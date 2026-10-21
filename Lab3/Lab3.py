import random
import matplotlib.pyplot as plt

class Hole:
    def __init__(self, x, y, lifetime):
        self.x = x
        self.y = y
        self.lifetime = lifetime
        self.active = True

    def tick(self):
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.active = False


class Tileworld:
    def __init__(self, size=15, spawn_prob=0.1, min_lifetime=20, max_lifetime=40):
        self.size = size
        self.spawn_prob = spawn_prob
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.holes = []
        self.holes_appeared = 0

    def tick(self):
        for h in self.holes:
            h.tick()
        self.holes = [h for h in self.holes if h.active]

        if random.random() < self.spawn_prob:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)

            if not any(h.x == x and h.y == y for h in self.holes):
                lifetime = random.randint(self.min_lifetime, self.max_lifetime)
                self.holes.append(Hole(x, y, lifetime))
                self.holes_appeared += 1


class BDIAgent:
    def __init__(self, x=7, y=7, gamma=1, commitment="single-minded", verbose=False):
        self.x = x
        self.y = y
        self.gamma = gamma
        self.commitment = commitment
        self.verbose = verbose
        self.beliefs = []
        self.desires = []
        self.intention = None
        self.plan = []
        self.s = 0
        self.holes_filled = 0
        self.events = []

    def log(self, message):
        self.events.append(message)
        if self.verbose:
            print(message)

    def brf(self, percepts):
        self.beliefs = list(percepts)

    def options(self):
        self.desires = list(self.beliefs)
        return self.desires

    def distance(self, hole):
        return abs(hole.x - self.x) + abs(hole.y - self.y)

    def filter(self, desires):
        if not desires:
            return None
        return min(
            desires,
            key=lambda h: (self.distance(h), -h.lifetime)
        )

    def generate_plan(self, target):
        if target is None:
            return []

        plan = []
        x, y = self.x, self.y

        while x != target.x or y != target.y:
            if x < target.x:
                plan.append((1, 0))
                x += 1
            elif x > target.x:
                plan.append((-1, 0))
                x -= 1
            elif y < target.y:
                plan.append((0, 1))
                y += 1
            elif y > target.y:
                plan.append((0, -1))
                y -= 1

        return plan

    def commit(self):
        choice = self.filter(self.options())

        if choice is not None:
            self.intention = choice
            self.plan = self.generate_plan(choice)

            self.log(
                f"EVENT: COMMITTED to ({choice.x}, {choice.y})"
            )

    def run_cycle(self, world, world_speed):
        self.brf(world.holes)

        if self.intention is not None:
            if (
                self.x == self.intention.x
                and self.y == self.intention.y
                and self.intention.active
            ):
                self.holes_filled += 1
                self.intention.active = False

                self.log(
                    f"EVENT: ACHIEVED ({self.x}, {self.y})"
                )

                self.intention = None
                self.plan = []
                self.s = 0

        if self.intention is not None:
            if self.intention not in self.beliefs:
                if self.commitment != "blind":
                    self.log(
                        f"EVENT: DROPPED "
                        f"({self.intention.x}, {self.intention.y})"
                    )

                    self.intention = None
                    self.plan = []
                    self.s = 0

        if self.intention is None:
            self.commit()

        elif not self.plan:
            self.commit()

        elif self.gamma != "bold" and self.s >= self.gamma:
            self.s = 0

            for _ in range(world_speed):
                world.tick()

            self.brf(world.holes)

            if self.commitment == "single-minded":

                if self.intention not in self.beliefs:
                    self.log(
                        f"EVENT: DROPPED during reconsideration "
                        f"({self.intention.x}, {self.intention.y})"
                    )

                    self.intention = None
                    self.plan = []

                    self.commit()

            elif self.commitment == "open-minded":

                new_intention = self.filter(
                    self.options()
                )

                if new_intention is None:
                    self.intention = None
                    self.plan = []

                elif new_intention != self.intention:
                    old = self.intention

                    self.intention = new_intention
                    self.plan = self.generate_plan(
                        new_intention
                    )

                    self.log(
                        f"EVENT: SWITCHED from "
                        f"({old.x}, {old.y}) to "
                        f"({new_intention.x}, {new_intention.y})"
                    )

        if self.plan:
            action = self.plan.pop(0)

            self.x += action[0]
            self.y += action[1]

            self.s += 1

        for _ in range(world_speed):
            world.tick()

def trace_demo():
    for seed in range(1000):
        random.seed(seed)

        world = Tileworld(
            size=15,
            spawn_prob=0.18,
            min_lifetime=8,
            max_lifetime=20
        )

        agent = BDIAgent(
            gamma=4,
            commitment="open-minded"
        )

        for _ in range(200):
            agent.run_cycle(world, 2)

        text = " ".join(agent.events)

        if all(x in text for x in ["COMMITTED", "ACHIEVED", "DROPPED", "SWITCHED"]):
            break

    print("\nBDI EVENT TRACE")
    print("-" * 50)

    random.seed(seed)

    world = Tileworld(
        size=15,
        spawn_prob=0.18,
        min_lifetime=8,
        max_lifetime=20
    )

    agent = BDIAgent(
        gamma=4,
        commitment="open-minded",
        verbose=True
    )

    for _ in range(200):
        agent.run_cycle(world, 2)

        if all(
            x in " ".join(agent.events)
            for x in ["COMMITTED", "ACHIEVED", "DROPPED", "SWITCHED"]
        ):
            break

    effectiveness = (
        agent.holes_filled / world.holes_appeared * 100
        if world.holes_appeared > 0 else 0
    )

    print("\nTrace Results")
    print(f"Seed: {seed}")
    print(f"Holes appeared: {world.holes_appeared}")
    print(f"Holes filled: {agent.holes_filled}")
    print(f"Effectiveness: {effectiveness:.2f}%")


def simulate(seed, gamma, speed):
    random.seed(seed)

    world = Tileworld(
        size=15,
        spawn_prob=0.1,
        min_lifetime=5,
        max_lifetime=15
    )

    agent = BDIAgent(
        gamma=gamma,
        commitment="single-minded"
    )

    for _ in range(600):
        agent.run_cycle(world, speed)

    if world.holes_appeared == 0:
        return 0

    return agent.holes_filled / world.holes_appeared * 100


def exercise_1():
    gammas = [1, 2, 4, 8, "bold"]
    speeds = [1, 2, 4, 8]
    seeds = range(25)

    results = {}

    for speed in speeds:
        results[speed] = {}

        for gamma in gammas:
            values = [
                simulate(seed, gamma, speed)
                for seed in seeds
            ]

            results[speed][gamma] = sum(values) / len(values)

    print("\nEXERCISE 1: BOLDNESS AGAINST DYNAMISM")
    print("-" * 70)

    print(
        f"{'Speed':<10}"
        f"{'γ=1':>12}"
        f"{'γ=2':>12}"
        f"{'γ=4':>12}"
        f"{'γ=8':>12}"
        f"{'Bold':>12}"
    )

    print("-" * 70)

    for speed in speeds:
        print(
            f"{speed:<10}"
            f"{results[speed][1]:>12.2f}"
            f"{results[speed][2]:>12.2f}"
            f"{results[speed][4]:>12.2f}"
            f"{results[speed][8]:>12.2f}"
            f"{results[speed]['bold']:>12.2f}"
        )

    print("\nBest γ at each world speed")
    print("-" * 40)

    best = {}

    for speed in speeds:
        best_gamma = max(
            results[speed],
            key=results[speed].get
        )

        best[speed] = best_gamma

        print(
            f"Speed {speed}: "
            f"γ = {best_gamma}, "
            f"Effectiveness = "
            f"{results[speed][best_gamma]:.2f}%"
        )

    plt.figure(figsize=(9, 6))

    for gamma in gammas:
        values = [
            results[speed][gamma]
            for speed in speeds
        ]

        plt.plot(
            speeds,
            values,
            marker="o",
            label=f"γ = {gamma}"
        )

    plt.xlabel("World ticks per agent action")
    plt.ylabel("Average effectiveness (%)")
    plt.title("Boldness Against World Dynamism")
    plt.xticks(speeds)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment3_exercise1.png", dpi=300)
    plt.show()

    return results, best


class ResourceBDIAgent:
    def __init__(
        self,
        x=7,
        y=7,
        battery_capacity=40,
        charger=(7, 7)
    ):
        self.x = x
        self.y = y
        self.battery_capacity = battery_capacity
        self.battery = battery_capacity
        self.charger = charger

        self.beliefs = []
        self.desires = []
        self.intention = None
        self.plan = []
        self.plan_type = None

        self.holes_filled = 0
        self.stranded_steps = 0

    def brf(self, percepts):
        self.beliefs = list(percepts)

    def distance(self, x, y):
        return abs(x - self.x) + abs(y - self.y)

    def distance_to_charger(self, x, y):
        return abs(x - self.charger[0]) + abs(y - self.charger[1])

    def resource_filter(self):
        feasible = []

        for hole in self.beliefs:
            if not hole.active:
                continue

            trip = self.distance(hole.x, hole.y)
            return_trip = self.distance_to_charger(hole.x, hole.y)

            if self.battery >= trip + return_trip:
                feasible.append(hole)

        if not feasible:
            return None

        return min(
            feasible,
            key=lambda h: self.distance(h.x, h.y)
        )

    def generate_plan(self, tx, ty):
        plan = []
        x, y = self.x, self.y

        while x != tx or y != ty:
            if x < tx:
                plan.append((1, 0))
                x += 1
            elif x > tx:
                plan.append((-1, 0))
                x -= 1
            elif y < ty:
                plan.append((0, 1))
                y += 1
            elif y > ty:
                plan.append((0, -1))
                y -= 1

        return plan

    def deliberate(self):
        if self.x == self.charger[0] and self.y == self.charger[1]:
            self.battery = self.battery_capacity

        hole = self.resource_filter()

        if hole is not None:
            self.intention = hole
            self.plan_type = "fill-hole"
            self.plan = self.generate_plan(
                hole.x,
                hole.y
            )
        else:
            self.intention = None
            self.plan_type = "recharge"
            self.plan = self.generate_plan(
                self.charger[0],
                self.charger[1]
            )

    def run_cycle(self, world):
        self.brf(world.holes)

        if self.intention is not None:
            if (
                self.x == self.intention.x
                and self.y == self.intention.y
            ):
                self.holes_filled += 1
                self.intention.active = False
                self.intention = None
                self.plan = []
                self.plan_type = None

        if not self.plan:
            self.deliberate()

        if self.plan:
            if self.battery > 0:
                action = self.plan.pop(0)

                self.x += action[0]
                self.y += action[1]

                self.battery -= 1
            else:
                self.stranded_steps += 1

        if (
            self.x == self.charger[0]
            and self.y == self.charger[1]
        ):
            self.battery = self.battery_capacity

        world.tick()


class BatteryBlindAgent:
    def __init__(self, x=7, y=7, battery_capacity=40):
        self.x = x
        self.y = y
        self.battery_capacity = battery_capacity
        self.battery = battery_capacity

        self.intention = None
        self.plan = []

        self.holes_filled = 0
        self.stranded_steps = 0

    def distance(self, hole):
        return abs(hole.x - self.x) + abs(hole.y - self.y)

    def generate_plan(self, hole):
        plan = []
        x, y = self.x, self.y

        while x != hole.x or y != hole.y:
            if x < hole.x:
                plan.append((1, 0))
                x += 1
            elif x > hole.x:
                plan.append((-1, 0))
                x -= 1
            elif y < hole.y:
                plan.append((0, 1))
                y += 1
            elif y > hole.y:
                plan.append((0, -1))
                y -= 1

        return plan

    def run_cycle(self, world):
        active_holes = [h for h in world.holes if h.active ]

        if self.intention is None or not self.intention.active:
            if active_holes:
                self.intention = min(
                    active_holes,
                    key=self.distance
                )
                self.plan = self.generate_plan(
                    self.intention
                )

        if self.intention is not None:
            if (
                self.x == self.intention.x
                and self.y == self.intention.y
                and self.intention.active
            ):
                self.holes_filled += 1
                self.intention.active = False
                self.intention = None
                self.plan = []

        if self.plan:
            if self.battery > 0:
                action = self.plan.pop(0)

                self.x += action[0]
                self.y += action[1]

                self.battery -= 1
            else:
                self.stranded_steps += 1

        world.tick()


def exercise_2():
    steps = 300
    seed = 100

    random.seed(seed)

    blind_world = Tileworld(
        size=15,
        spawn_prob=0.20,
        min_lifetime=20,
        max_lifetime=40
    )

    blind_agent = BatteryBlindAgent()

    for _ in range(steps):
        blind_agent.run_cycle(blind_world)

    random.seed(seed)

    smart_world = Tileworld(
        size=15,
        spawn_prob=0.20,
        min_lifetime=20,
        max_lifetime=40
    )

    smart_agent = ResourceBDIAgent(
        battery_capacity=40,
        charger=(7, 7)
    )

    for _ in range(steps):
        smart_agent.run_cycle(smart_world)

    print("\nEXERCISE 2: RESOURCE-BOUNDED FILTER")
    print("-" * 60)

    print(
        f"{'Agent':<25}"
        f"{'Holes Filled':>15}"
        f"{'Stranded Steps':>20}"
    )

    print("-" * 60)

    print(
        f"{'Battery-Blind':<25}"
        f"{blind_agent.holes_filled:>15}"
        f"{blind_agent.stranded_steps:>20}"
    )

    print(
        f"{'Resource-Bounded BDI':<25}"
        f"{smart_agent.holes_filled:>15}"
        f"{smart_agent.stranded_steps:>20}"
    )

    print("\nBDI component responsible: FILTER")

    print(
        "The filter checks whether the battery is sufficient "
        "for the trip to the hole and the return to the charger. "
        "The planner only constructs the plan after an intention "
        "has been selected, so the resource constraint belongs "
        "in deliberation/filtering rather than planning."
    )


if __name__ == "__main__":
    trace_demo()
    results, best_gamma = exercise_1()
    exercise_2()