
"""Experiment 12: Project RESCUE-SWARM
Cooperative Multi-Agent Reinforcement Learning with tabular Q-learning.

Compares:
  1) Independent Q-learning: each rescue agent has its own Q-table and receives
     its own local reward and partial local state.
  2) Cooperative Q-learning: agents use a shared global state and one joint-action
     Q-table, and learn from a common team reward.


Outputs (in the current working directory):
    lab12_u23ai093_results.png
    lab12_u23ai093_metrics.csv
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Position = Tuple[int, int]
Action = int

# Action indices: stay, up, down, left, right
ACTIONS: Tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("Stay", "Up", "Down", "Left", "Right")
FIXED_VICTIMS: Tuple[Position, ...] = ((0, 6), (3, 1), (6, 5))


@dataclass
class StepInfo:
    rescued_this_step: int
    collisions_this_step: int
    hazard_entries_this_step: int
    invalid_moves_this_step: int
    completed: bool


class RescueGridWorld:
    """Small fully observable grid world used for the learning experiment."""

    def __init__(self, size: int = 7, max_steps: int = 100, seed: int = 0):
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.starts: Tuple[Position, Position] = ((0, 0), (6, 0))
        self.obstacles = {
            (1, 2), (2, 2), (4, 3), (5, 3), (3, 4), (3, 2)
        }
        self.hazards = {(1, 4), (2, 4), (5, 1), (5, 2)}
        self.victim_candidates = [
            (0, 6), (3, 1), (6, 5), (2, 6), (4, 0),
            (6, 6), (0, 3), (3, 6), (4, 5), (1, 5),
        ]
        self._validate_layout()
        self.victim_bit = {p: (1 << i) for i, p in enumerate(self.victim_candidates)}
        self.reset()

    def _validate_layout(self) -> None:
        for p in self.starts:
            if p in self.obstacles:
                raise ValueError(f"Start cell {p} cannot be an obstacle")
        self.victim_candidates = [
            p for p in self.victim_candidates
            if p not in self.obstacles and p not in self.starts
        ]
        if len(self.victim_candidates) < 3:
            raise ValueError("At least three valid victim locations are required")

    def reset(self, seed: int | None = None, victims: Sequence[Position] | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.agent_positions: List[Position] = [self.starts[0], self.starts[1]]
        if victims is None:
            self.victims = set(self.rng.sample(self.victim_candidates, 3))
        else:
            selected = list(victims)
            if len(selected) == 0 or len(set(selected)) != len(selected):
                raise ValueError("Victims must be a non-empty set of unique cells")
            for p in selected:
                if p in self.obstacles or not self._inside(p):
                    raise ValueError(f"Invalid victim location: {p}")
            self.victims = set(selected)
        self.initial_victim_count = len(self.victims)
        self.steps = 0
        self.total_collisions = 0
        self.total_hazard_entries = 0
        self.total_invalid_moves = 0
        return self.get_observations()

    def _inside(self, p: Position) -> bool:
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def _clip_delta(self, v: int) -> int:
        return max(-self.size + 1, min(self.size - 1, v))

    def _nearest_victim_delta(self, pos: Position) -> Position:
        if not self.victims:
            return (0, 0)
        target = min(self.victims, key=lambda v: (abs(v[0] - pos[0]) + abs(v[1] - pos[1]), v[0], v[1]))
        return (self._clip_delta(target[0] - pos[0]), self._clip_delta(target[1] - pos[1]))

    def _nearest_victim_distance(self, pos: Position, victims: set[Position]) -> int:
        if not victims:
            return 0
        return min(abs(v[0] - pos[0]) + abs(v[1] - pos[1]) for v in victims)

    def get_observations(self) -> List[tuple]:
        """Return one compact observation per agent.

        Each observation contains absolute position, nearest-victim direction,
        nearby blocked/hazard directions, remaining-victim count, teammate's
        relative position, and a bitmask of all remaining victims. Independent
        learners omit teammate information; the cooperative learner uses a
        centralized joint state assembled from both agent positions and victim mask.
        """
        observations = []
        for agent_id, pos in enumerate(self.agent_positions):
            target_dr, target_dc = self._nearest_victim_delta(pos)
            blocked_mask = 0
            hazard_mask = 0
            # Direction bits follow Up, Down, Left, Right.
            directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
            for bit, (dr, dc) in enumerate(directions):
                nxt = (pos[0] + dr, pos[1] + dc)
                if not self._inside(nxt) or nxt in self.obstacles:
                    blocked_mask |= (1 << bit)
                if nxt in self.hazards:
                    hazard_mask |= (1 << bit)
            partner = self.agent_positions[1 - agent_id]
            partner_dr = self._clip_delta(partner[0] - pos[0])
            partner_dc = self._clip_delta(partner[1] - pos[1])
            observations.append((
                agent_id,
                pos[0], pos[1],
                target_dr, target_dc,
                blocked_mask, hazard_mask,
                len(self.victims),
                partner_dr, partner_dc,
                sum(self.victim_bit[v] for v in self.victims),
            ))
        return observations

    def step(self, actions: Sequence[Action]):
        if len(actions) != 2 or any(a < 0 or a >= len(ACTIONS) for a in actions):
            raise ValueError("Exactly two valid actions are required")

        self.steps += 1
        old_positions = list(self.agent_positions)
        old_victims = set(self.victims)
        old_target_distances = [self._nearest_victim_distance(p, old_victims) for p in old_positions]
        proposed = []
        local_rewards = [-1.0, -1.0]  # movement cost: discourages unnecessary steps
        invalid = [False, False]
        hazard_entries = [False, False]

        # First calculate moves, including boundary/obstacle handling.
        for i, action in enumerate(actions):
            dr, dc = ACTIONS[action]
            nxt = (old_positions[i][0] + dr, old_positions[i][1] + dc)
            if not self._inside(nxt) or nxt in self.obstacles:
                nxt = old_positions[i]
                invalid[i] = True
                local_rewards[i] -= 2.0
            elif nxt in self.hazards and nxt != old_positions[i]:
                hazard_entries[i] = True
                local_rewards[i] -= 6.0
            proposed.append(nxt)

        # Resolve vertex and edge conflicts. Both agents remain in place.
        collision = [False, False]
        same_cell = proposed[0] == proposed[1]
        swap = proposed[0] == old_positions[1] and proposed[1] == old_positions[0]
        if same_cell or swap:
            collision = [True, True]
            proposed = old_positions.copy()
            local_rewards[0] -= 5.0
            local_rewards[1] -= 5.0

        self.agent_positions = proposed

        # Rescue a victim when an agent reaches its cell.
        rescued_by_agent = [0, 0]
        for i, pos in enumerate(self.agent_positions):
            if pos in self.victims:
                self.victims.remove(pos)
                rescued_by_agent[i] += 1
                local_rewards[i] += 25.0

        # Small progress shaping encourages movement toward remaining victims.
        for i, pos in enumerate(self.agent_positions):
            new_distance = self._nearest_victim_distance(pos, self.victims)
            local_rewards[i] += 0.4 * (old_target_distances[i] - new_distance)

        completed = len(self.victims) == 0
        if completed:
            # One completion bonus, assigned to the agent that performed the final rescue.
            final_rescuer = 0 if rescued_by_agent[0] else 1
            local_rewards[final_rescuer] += 20.0

        self.total_collisions += int(any(collision))
        self.total_hazard_entries += sum(hazard_entries)
        self.total_invalid_moves += sum(invalid)

        done = completed or self.steps >= self.max_steps
        info = StepInfo(
            rescued_this_step=sum(rescued_by_agent),
            collisions_this_step=int(any(collision)),
            hazard_entries_this_step=sum(hazard_entries),
            invalid_moves_this_step=sum(invalid),
            completed=completed,
        )
        return self.get_observations(), np.asarray(local_rewards, dtype=float), done, info

    def render_ascii(self) -> str:
        """Text rendering useful for debugging or lab observation."""
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                p = (r, c)
                if p == self.agent_positions[0]:
                    cell = "A"
                elif p == self.agent_positions[1]:
                    cell = "B"
                elif p in self.victims:
                    cell = "V"
                elif p in self.obstacles:
                    cell = "#"
                elif p in self.hazards:
                    cell = "H"
                else:
                    cell = "."
                cells.append(cell)
            rows.append(" ".join(cells))
        return "\n".join(rows)


class TabularQLearner:
    def __init__(self, n_actions: int = len(ACTIONS), alpha: float = 0.25, gamma: float = 0.95):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def choose_action(self, state: tuple, epsilon: float, rng: random.Random) -> int:
        if rng.random() < epsilon:
            return rng.randrange(self.n_actions)
        values = self.q[state]
        best = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(rng.choice(best.tolist()))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> None:
        old_value = self.q[state][action]
        future = 0.0 if done else float(np.max(self.q[next_state]))
        target = reward + self.gamma * future
        self.q[state][action] = old_value + self.alpha * (target - old_value)


def independent_state(obs: tuple) -> tuple:
    """Independent learner uses its own view and the shared remaining-victim mask."""
    return obs[:8] + (obs[10],)


def epsilon_for_episode(ep: int, total_episodes: int, start: float = 1.0, end: float = 0.05) -> float:
    fraction = ep / max(1, total_episodes - 1)
    return start + (end - start) * fraction


def train_independent(episodes: int = 1800, seed: int = 11):
    env = RescueGridWorld(seed=seed)
    learners = [TabularQLearner(), TabularQLearner()]
    rng = random.Random(seed + 100)
    rewards_curve = []
    for ep in range(episodes):
        obs = env.reset(victims=FIXED_VICTIMS)
        epsilon = epsilon_for_episode(ep, episodes)
        episode_return = 0.0
        done = False
        while not done:
            states = [independent_state(o) for o in obs]
            actions = [learners[i].choose_action(states[i], epsilon, rng) for i in range(2)]
            next_obs, local_rewards, done, _ = env.step(actions)
            next_states = [independent_state(o) for o in next_obs]
            for i in range(2):
                learners[i].update(states[i], actions[i], float(local_rewards[i]), next_states[i], done)
            episode_return += float(np.sum(local_rewards))
            obs = next_obs
        rewards_curve.append(episode_return)
    return learners, np.asarray(rewards_curve, dtype=float)


def joint_state(env: RescueGridWorld) -> tuple:
    """Central shared state: both agent positions and the complete remaining-victim map."""
    victim_mask = sum(env.victim_bit[v] for v in env.victims)
    a0, a1 = env.agent_positions
    return (a0[0], a0[1], a1[0], a1[1], victim_mask)


def train_cooperative(episodes: int = 15000, seed: int = 29):
    """Centralized cooperative Q-learning over joint states and joint actions.

    One Q-table selects a pair of actions (5 x 5 possibilities) using the full
    shared state. Both agents receive the common reward equal to the sum of
    local rewards. This is practical for this deliberately small grid world.
    """
    env = RescueGridWorld(seed=seed)
    shared_learner = TabularQLearner(n_actions=len(ACTIONS) ** 2)
    rng = random.Random(seed + 100)
    rewards_curve = []
    for ep in range(episodes):
        env.reset(victims=FIXED_VICTIMS)
        state = joint_state(env)
        epsilon = epsilon_for_episode(ep, episodes)
        episode_return = 0.0
        done = False
        while not done:
            joint_action = shared_learner.choose_action(state, epsilon, rng)
            actions = (joint_action // len(ACTIONS), joint_action % len(ACTIONS))
            _, local_rewards, done, _ = env.step(actions)
            team_reward = float(np.sum(local_rewards))
            next_state = joint_state(env)
            shared_learner.update(state, joint_action, team_reward, next_state, done)
            episode_return += team_reward
            state = next_state
        rewards_curve.append(episode_return)
    return shared_learner, np.asarray(rewards_curve, dtype=float)


def evaluate_policy(policy_type: str, learners, episodes: int = 40, seed: int = 5000):
    env = RescueGridWorld(seed=seed)
    rng = random.Random(seed + 1)
    records = []
    for ep in range(episodes):
        # Both methods are evaluated on the same fixed disaster scenario.
        obs = env.reset(victims=FIXED_VICTIMS)
        episode_return = 0.0
        done = False
        while not done:
            if policy_type == "independent":
                actions = [
                    learners[i].choose_action(independent_state(obs[i]), 0.0, rng)
                    for i in range(2)
                ]
            elif policy_type == "cooperative":
                combo = learners.choose_action(joint_state(env), 0.0, rng)
                actions = (combo // len(ACTIONS), combo % len(ACTIONS))
            else:
                raise ValueError("Unknown policy_type")
            obs, local_rewards, done, _ = env.step(actions)
            episode_return += float(np.sum(local_rewards))

        rescued = env.initial_victim_count - len(env.victims)
        records.append({
            "rescued": rescued,
            "completion_time": env.steps,
            "completed": int(len(env.victims) == 0),
            "cumulative_reward": episode_return,
            "collisions": env.total_collisions,
            "hazard_entries": env.total_hazard_entries,
            "invalid_moves": env.total_invalid_moves,
        })
    return records


def summarize(records: Sequence[dict]) -> dict:
    keys = records[0].keys()
    return {key: float(np.mean([row[key] for row in records])) for key in keys}


def moving_average(values: np.ndarray, window: int = 60) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def run_demonstration(policy_type: str, learners, seed: int = 9001):
    env = RescueGridWorld(seed=seed)
    obs = env.reset(victims=FIXED_VICTIMS)
    initial_victims = set(env.victims)
    paths = [[env.agent_positions[0]], [env.agent_positions[1]]]
    rescued_events = []
    done = False
    rng = random.Random(seed + 7)
    while not done:
        if policy_type == "independent":
            actions = [
                learners[i].choose_action(independent_state(obs[i]), 0.0, rng)
                for i in range(2)
            ]
        else:
            combo = learners.choose_action(joint_state(env), 0.0, rng)
            actions = (combo // len(ACTIONS), combo % len(ACTIONS))
        obs, _, done, info = env.step(actions)
        for i in range(2):
            paths[i].append(env.agent_positions[i])
        if info.rescued_this_step:
            rescued_events.append((env.steps, info.rescued_this_step))
    return env, paths, initial_victims, rescued_events


def plot_results(ind_rewards, coop_rewards, ind_eval, coop_eval, output_path: Path,
                 demo_paths, demo_victims, grid_size: int = 7,
                 obstacles=None, hazards=None, starts=None):
    obstacles = obstacles or set()
    hazards = hazards or set()
    starts = starts or ((0, 0), (6, 0))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # Learning curves.
    window = 60
    ind_ma = moving_average(ind_rewards, window)
    coop_ma = moving_average(coop_rewards, window)
    offset = window - 1 if len(ind_rewards) >= window else 0
    axes[0].plot(np.arange(len(ind_ma)) + offset, ind_ma, label="Independent Q-learning")
    axes[0].plot(np.arange(len(coop_ma)) + offset, coop_ma, label="Cooperative Q-learning")
    axes[0].set_title("Training learning curves")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative episode reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Evaluation metric comparison. Values are normalized separately by metric;
    # raw means are written to the CSV file and printed to the console.
    labels = ["Rescues", "Steps", "Reward", "Collisions"]
    metrics = ["rescued", "completion_time", "cumulative_reward", "collisions"]
    x = np.arange(len(labels))
    width = 0.36
    ind_vals = [ind_eval[m] for m in metrics]
    coop_vals = [coop_eval[m] for m in metrics]
    norm_ind, norm_coop = [], []
    for a, b in zip(ind_vals, coop_vals):
        scale = max(abs(a), abs(b), 1.0)
        norm_ind.append(a / scale)
        norm_coop.append(b / scale)
    axes[1].bar(x - width / 2, norm_ind, width, label="Independent")
    axes[1].bar(x + width / 2, norm_coop, width, label="Cooperative")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(min(-1.1, min(norm_ind + norm_coop) - 0.1), 1.15)
    axes[1].set_title("Evaluation comparison")
    axes[1].set_ylabel("Normalized mean (raw values in CSV)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    # One cooperative episode showing agent routes and rescue locations.
    ax = axes[2]
    for r, c in obstacles:
        ax.scatter(c, r, marker="s", s=115, label="Obstacle" if (r, c) == sorted(obstacles)[0] else None)
    for r, c in hazards:
        ax.scatter(c, r, marker="X", s=100, label="Hazard" if (r, c) == sorted(hazards)[0] else None)
    for r, c in demo_victims:
        ax.scatter(c, r, marker="*", s=180, label="Victim" if (r, c) == sorted(demo_victims)[0] else None)
    for i, path in enumerate(demo_paths):
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, marker="o", markersize=3, linewidth=1.8, label=f"Rescue agent {i + 1}")
        ax.scatter(cols[0], rows[0], marker="s", s=70)
        ax.scatter(cols[-1], rows[-1], marker="D", s=55)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)
    ax.set_title("Cooperative rescue routes")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle("Project RESCUE-SWARM: Independent vs Cooperative Multi-Agent Q-Learning", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path: Path, ind_summary: dict, coop_summary: dict) -> None:
    fieldnames = ["approach", "rescued", "completion_time", "completed", "cumulative_reward", "collisions", "hazard_entries", "invalid_moves"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"approach": "Independent Q-learning", **ind_summary})
        writer.writerow({"approach": "Cooperative Q-learning", **coop_summary})


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Average successful rescues : {summary['rescued']:.2f} / 3")
    print(f"Average completion time    : {summary['completion_time']:.2f} steps")
    print(f"Completion rate            : {summary['completed'] * 100:.1f}%")
    print(f"Average cumulative reward  : {summary['cumulative_reward']:.2f}")
    print(f"Average collisions         : {summary['collisions']:.2f}")
    print(f"Average hazard entries     : {summary['hazard_entries']:.2f}")
    print(f"Average invalid moves      : {summary['invalid_moves']:.2f}")


def main():
    random.seed(42)
    np.random.seed(42)
    output_dir = Path(__file__).resolve().parent
    episodes = 2000
    evaluation_episodes = 40

    print("PROJECT RESCUE-SWARM - Cooperative Multi-Agent Q-Learning")
    print(f"Training episodes per approach: {episodes}")
    print("Training independent agents...")
    independent_learners, independent_rewards = train_independent(episodes=episodes, seed=11)
    print("Training cooperative agents...")
    cooperative_learner, cooperative_rewards = train_cooperative(episodes=episodes, seed=29)

    print("Evaluating learned policies on the same 40 scenarios...")
    independent_records = evaluate_policy("independent", independent_learners, episodes=evaluation_episodes, seed=6000)
    cooperative_records = evaluate_policy("cooperative", cooperative_learner, episodes=evaluation_episodes, seed=6000)
    independent_summary = summarize(independent_records)
    cooperative_summary = summarize(cooperative_records)

    print_summary("INDEPENDENT Q-LEARNING", independent_summary)
    print_summary("COOPERATIVE Q-LEARNING", cooperative_summary)

    plot_path = output_dir / "lab12_u23ai093_results.png"
    csv_path = output_dir / "lab12_u23ai093_metrics.csv"
    demo_env, demo_paths, demo_victims, _ = run_demonstration("cooperative", cooperative_learner, seed=9001)
    plot_results(
        independent_rewards, cooperative_rewards, independent_summary, cooperative_summary,
        plot_path, demo_paths, demo_victims, grid_size=demo_env.size,
        obstacles=demo_env.obstacles, hazards=demo_env.hazards, starts=demo_env.starts,
    )
    write_metrics_csv(csv_path, independent_summary, cooperative_summary)

    print("\nOBSERVATIONS")
    print("1. Independent Q-learning updates each agent using its own reward and local observation.")
    print("2. Cooperative Q-learning uses a shared global state, one joint-action Q-table, and a common team reward.")
    print("3. Compare completion rate, rescue count, completion time, reward, and collisions using the evaluation results above.")
    print("4. The cooperative controller can coordinate both agents jointly; independent agents select actions separately and may duplicate effort.")
    print("5. The grid includes obstacles and hazardous cells. Invalid moves and collisions are penalized; entering a hazard incurs an additional penalty.")
    print(f"\nSaved visualization: {plot_path}")
    print(f"Saved evaluation metrics: {csv_path}")

    # One reproducible demonstration episode, with a concise final map snapshot.
    print("\nCOOPERATIVE DEMONSTRATION SCENARIO")
    print(f"Initial victims: {sorted(demo_victims)}")
    print(f"Rescued: {demo_env.initial_victim_count - len(demo_env.victims)} / {demo_env.initial_victim_count}")
    print(f"Steps: {demo_env.steps}; collisions: {demo_env.total_collisions}; hazard entries: {demo_env.total_hazard_entries}")
    print(demo_env.render_ascii())


if __name__ == "__main__":
    main()
