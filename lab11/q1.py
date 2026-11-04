
"""Experiment 11 - Project LAST EXIT: Intelligent Disaster Evacuation System

Implements:
1. Knowledge representation with entity objects and relationship triples.
2. Forward-chaining IF-THEN rules for priority and rescue actions.
3. Fuzzy reasoning for uncertain flood, road, and report information.
4. Resource allocation for rescue teams, fuel/trips, and shelter capacity.
5. Contract Net Protocol (CNP) for multi-agent task bidding and awards.
6. Baseline and disrupted simulations, metrics, and result visualization.


The simulation is illustrative. Its times, populations, road conditions, capacities,
and event are synthetic values chosen for this laboratory exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from heapq import heappop, heappush
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any


# -----------------------------------------------------------------------------
# 1. KNOWLEDGE REPRESENTATION: ENTITIES AND RELATIONSHIPS
# -----------------------------------------------------------------------------

@dataclass
class AffectedArea:
    area_id: str
    name: str
    node: str
    population: int
    severity: int                 # Integer from 1 (low) to 5 (critical)
    flood_level: float            # Uncertain level in [0, 1]
    report_confidence: float      # Reliability of reports in [0, 1]


@dataclass
class Shelter:
    shelter_id: str
    name: str
    node: str
    capacity: int
    open: bool = True
    occupied: int = 0

    @property
    def remaining_capacity(self) -> int:
        if not self.open:
            return 0
        return max(0, self.capacity - self.occupied)


@dataclass
class RescueTeam:
    team_id: str
    name: str
    base_node: str
    seats_per_trip: int
    max_trips: int
    fuel_available: int
    equipment: Set[str]
    available: bool = True

    @property
    def total_people_capacity(self) -> int:
        return self.seats_per_trip * min(self.max_trips, self.fuel_available)


@dataclass
class RoadEdge:
    destination: str
    travel_time: float
    quality: float               # 0 (very poor) to 1 (excellent)
    open: bool = True


class RoadNetwork:
    """Undirected road graph. Dijkstra uses travel time adjusted by road quality."""

    def __init__(self) -> None:
        self.graph: Dict[str, List[RoadEdge]] = {}

    def add_road(self, a: str, b: str, travel_time: float,
                 quality: float = 0.9) -> None:
        quality = max(0.0, min(1.0, quality))
        self.graph.setdefault(a, []).append(RoadEdge(b, travel_time, quality))
        self.graph.setdefault(b, []).append(RoadEdge(a, travel_time, quality))

    def block_road(self, a: str, b: str) -> None:
        found = False
        for edge in self.graph.get(a, []):
            if edge.destination == b:
                edge.open = False
                found = True
        for edge in self.graph.get(b, []):
            if edge.destination == a:
                edge.open = False
                found = True
        if not found:
            raise ValueError(f"Cannot block unknown road: {a} <-> {b}")

    @staticmethod
    def effective_time(edge: RoadEdge) -> float:
        # Poor road quality increases estimated travel time.
        return edge.travel_time * (1.0 + 0.5 * (1.0 - edge.quality))

    def shortest_path(self, start: str, goal: str
                      ) -> Optional[Tuple[float, List[str], float]]:
        """Return (adjusted travel time, path, average road quality), or None."""
        if start == goal:
            return 0.0, [start], 1.0
        if start not in self.graph or goal not in self.graph:
            return None

        distances: Dict[str, float] = {start: 0.0}
        previous: Dict[str, str] = {}
        heap: List[Tuple[float, str]] = [(0.0, start)]

        while heap:
            current_distance, node = heappop(heap)
            if current_distance > distances.get(node, float("inf")):
                continue
            if node == goal:
                break
            for edge in self.graph.get(node, []):
                if not edge.open:
                    continue
                candidate = current_distance + self.effective_time(edge)
                if candidate < distances.get(edge.destination, float("inf")):
                    distances[edge.destination] = candidate
                    previous[edge.destination] = node
                    heappush(heap, (candidate, edge.destination))

        if goal not in distances:
            return None

        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()

        qualities: List[float] = []
        for left, right in zip(path, path[1:]):
            for edge in self.graph[left]:
                if edge.destination == right and edge.open:
                    qualities.append(edge.quality)
                    break
        average_quality = sum(qualities) / len(qualities) if qualities else 1.0
        return distances[goal], path, average_quality


@dataclass
class KnowledgeBase:
    entities: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    facts: Set[Tuple[Any, ...]] = field(default_factory=set)

    def add_entity(self, entity_id: str, entity: Any) -> None:
        self.entities[entity_id] = entity

    def add_relation(self, subject: str, relation: str, object_id: str) -> None:
        triple = (subject, relation, object_id)
        if triple not in self.relations:
            self.relations.append(triple)

    def add_fact(self, *fact: Any) -> bool:
        fact_tuple = tuple(fact)
        if fact_tuple in self.facts:
            return False
        self.facts.add(fact_tuple)
        return True


# -----------------------------------------------------------------------------
# 2. FUZZY LOGIC AND FORWARD-CHAINING RULES
# -----------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """Triangular fuzzy membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def fuzzy_low(x: float) -> float:
    return clamp((0.50 - x) / 0.50)


def fuzzy_medium(x: float) -> float:
    return triangular_membership(x, 0.20, 0.50, 0.80)


def fuzzy_high(x: float) -> float:
    return clamp((x - 0.50) / 0.40)


def fuzzy_road_poor(road_risk: float) -> float:
    return clamp((road_risk - 0.15) / 0.65)


def estimate_fuzzy_risk(area: AffectedArea, roads: RoadNetwork,
                        shelters: List[Shelter]) -> Dict[str, float]:
    """Combine fuzzy flood, report uncertainty, and road-condition evidence."""
    flood_high = fuzzy_high(area.flood_level)
    flood_medium = fuzzy_medium(area.flood_level)
    report_uncertainty = 1.0 - area.report_confidence

    route_qualities: List[float] = []
    for shelter in shelters:
        if shelter.open:
            result = roads.shortest_path(area.node, shelter.node)
            if result is not None:
                route_qualities.append(result[2])
    average_quality = (max(route_qualities) if route_qualities else 0.0)
    road_risk = 1.0 - average_quality
    road_poor = fuzzy_road_poor(road_risk)

    # Fuzzy OR is represented by max(). The terms model uncertain evidence.
    overall_risk = max(flood_high, 0.70 * flood_medium,
                       0.75 * report_uncertainty, road_poor)
    return {
        "flood_low": fuzzy_low(area.flood_level),
        "flood_medium": flood_medium,
        "flood_high": flood_high,
        "report_uncertainty": report_uncertainty,
        "road_risk": road_risk,
        "road_poor": road_poor,
        "overall_risk": clamp(overall_risk),
    }


def forward_chain(kb: KnowledgeBase,
                  areas: List[AffectedArea],
                  risk_by_area: Dict[str, Dict[str, float]]) -> None:
    """Apply a small production-rule system until no new facts are generated."""
    for area in areas:
        kb.add_fact("affected", area.area_id)
        kb.add_fact("population", area.area_id, area.population)
        kb.add_fact("severity", area.area_id, area.severity)
        kb.add_fact("flood_level", area.area_id, area.flood_level)
        kb.add_fact("report_confidence", area.area_id, area.report_confidence)

    changed = True
    while changed:
        changed = False
        for area in areas:
            risk = risk_by_area[area.area_id]["overall_risk"]
            flood_high = risk_by_area[area.area_id]["flood_high"]

            # Rule 1: IF severity is critical OR estimated risk is very high,
            # THEN mark the area as critical priority.
            if area.severity >= 5 or risk >= 0.75:
                changed |= kb.add_fact("priority", area.area_id, "CRITICAL")
            # Rule 2: IF severity is high, THEN mark high priority.
            elif area.severity >= 4 or risk >= 0.50:
                changed |= kb.add_fact("priority", area.area_id, "HIGH")
            else:
                changed |= kb.add_fact("priority", area.area_id, "NORMAL")

            # Rule 3: IF flood membership is high, THEN a boat is required.
            if flood_high >= 0.55:
                changed |= kb.add_fact("requires_equipment", area.area_id, "boat")
                changed |= kb.add_fact("rescue_action", area.area_id,
                                       "deploy_boat_and_evacuate")
            else:
                changed |= kb.add_fact("rescue_action", area.area_id,
                                       "ground_transport_evacuation")

            # Rule 4: IF report confidence is low, THEN request verification.
            if area.report_confidence < 0.70:
                changed |= kb.add_fact("rescue_action", area.area_id,
                                       "verify_report_while_preparing_rescue")

            # Rule 5: IF population is large, THEN request multiple transport trips.
            if area.population > 45:
                changed |= kb.add_fact("resource_action", area.area_id,
                                       "plan_multiple_trips")


def calculate_priority_score(area: AffectedArea,
                             fuzzy_values: Dict[str, float]) -> float:
    severity_component = (area.severity / 5.0) * 0.55
    risk_component = fuzzy_values["overall_risk"] * 0.35
    uncertainty_component = fuzzy_values["report_uncertainty"] * 0.10
    return 100.0 * (severity_component + risk_component + uncertainty_component)


# -----------------------------------------------------------------------------
# 3, 4, 5. TEAM AGENTS, RESOURCE ALLOCATION, AND CONTRACT NET PROTOCOL
# -----------------------------------------------------------------------------

@dataclass
class ShelterAllocation:
    shelter_id: str
    people: int
    one_way_time: float
    path: List[str]
    trips: int


@dataclass
class TeamBid:
    team_id: str
    bid_value: float
    base_to_area_time: float
    base_to_area_path: List[str]
    shelter_allocations: List[ShelterAllocation]
    total_trips: int
    estimated_mission_time: float


class RescueTeamAgent:
    """A rescue-team agent that can evaluate a CNP task announcement."""

    def __init__(self, team: RescueTeam) -> None:
        self.team = team

    def calculate_bid(self, area: AffectedArea, roads: RoadNetwork,
                      shelters: List[Shelter],
                      required_equipment: Set[str]) -> Optional[TeamBid]:
        team = self.team
        if not team.available:
            return None
        if not required_equipment.issubset(team.equipment):
            return None
        if team.total_people_capacity < area.population:
            return None

        base_route = roads.shortest_path(team.base_node, area.node)
        if base_route is None:
            return None
        base_time, base_path, _ = base_route

        remaining = area.population
        shelter_allocations: List[ShelterAllocation] = []
        # Prefer reachable shelters with the lowest adjusted travel time.
        shelter_candidates = []
        for shelter in shelters:
            if shelter.remaining_capacity <= 0:
                continue
            route = roads.shortest_path(area.node, shelter.node)
            if route is not None:
                shelter_candidates.append((route[0], shelter.shelter_id,
                                           shelter, route[1]))
        shelter_candidates.sort(key=lambda item: item[0])

        for one_way_time, _, shelter, path in shelter_candidates:
            if remaining <= 0:
                break
            people = min(remaining, shelter.remaining_capacity)
            if people <= 0:
                continue
            trips = ceil(people / team.seats_per_trip)
            shelter_allocations.append(
                ShelterAllocation(shelter.shelter_id, people,
                                  one_way_time, path, trips)
            )
            remaining -= people

        # The bid is infeasible if all people cannot be placed in open shelters.
        if remaining > 0 or not shelter_allocations:
            return None

        total_trips = sum(item.trips for item in shelter_allocations)
        if total_trips > team.fuel_available or total_trips > team.max_trips:
            return None

        # Each trip carries people from the affected area to a shelter.
        # The team returns to the area between trips, except after the last trip.
        trip_times: List[float] = []
        for allocation in shelter_allocations:
            trip_times.extend([allocation.one_way_time] * allocation.trips)
        transport_time = sum(trip_times)
        if len(trip_times) > 1:
            transport_time += sum(trip_times[:-1])
        mission_time = base_time + transport_time

        # Lower bids are preferred. This balances response time and resource use.
        bid_value = (0.35 * base_time + 0.55 * mission_time
                     + 0.50 * total_trips)
        return TeamBid(
            team_id=team.team_id,
            bid_value=bid_value,
            base_to_area_time=base_time,
            base_to_area_path=base_path,
            shelter_allocations=shelter_allocations,
            total_trips=total_trips,
            estimated_mission_time=mission_time,
        )


class ContractNetCoordinator:
    """Task manager: announces tasks, collects bids, and awards contracts."""

    def __init__(self, teams: List[RescueTeam]) -> None:
        self.agents = [RescueTeamAgent(team) for team in teams]

    def announce_and_award(self, area: AffectedArea, roads: RoadNetwork,
                           shelters: List[Shelter],
                           required_equipment: Set[str]
                           ) -> Tuple[Optional[TeamBid], List[TeamBid]]:
        bids: List[TeamBid] = []
        for agent in self.agents:
            bid = agent.calculate_bid(area, roads, shelters,
                                      required_equipment)
            if bid is not None:
                bids.append(bid)
        bids.sort(key=lambda item: item.bid_value)
        winner = bids[0] if bids else None
        return winner, bids


def commit_award(bid: TeamBid, teams: List[RescueTeam],
                 shelters: List[Shelter]) -> None:
    """Reserve the winning team's fuel and the awarded shelter capacity."""
    team = next(team for team in teams if team.team_id == bid.team_id)
    team.available = False
    team.fuel_available -= bid.total_trips

    for allocation in bid.shelter_allocations:
        shelter = next(s for s in shelters
                       if s.shelter_id == allocation.shelter_id)
        if allocation.people > shelter.remaining_capacity:
            raise RuntimeError("Shelter capacity changed before award commit")
        shelter.occupied += allocation.people


# -----------------------------------------------------------------------------
# 6. SIMULATION SCENARIOS AND EVALUATION
# -----------------------------------------------------------------------------

def build_initial_state() -> Tuple[RoadNetwork, List[AffectedArea],
                                   List[Shelter], List[RescueTeam], KnowledgeBase]:
    roads = RoadNetwork()

    # Road network: travel-time values are illustrative minutes.
    road_data = [
        ("Command", "NorthHub", 5, 0.95),
        ("Command", "MarketHub", 4, 0.92),
        ("Command", "SouthHub", 6, 0.90),
        ("Command", "WestHub", 4, 0.88),
        ("NorthHub", "North", 3, 0.92),
        ("NorthHub", "Riverside", 7, 0.82),
        ("North", "Market", 4, 0.88),
        ("North", "HighSchool", 5, 0.90),
        ("North", "CentralSchool", 6, 0.86),
        ("MarketHub", "Market", 3, 0.92),
        ("Market", "CentralSchool", 4, 0.90),
        ("Market", "HighSchool", 5, 0.92),
        ("Market", "Stadium", 7, 0.85),
        ("Market", "WestHub", 4, 0.88),
        ("SouthHub", "Riverside", 3, 0.91),
        ("Riverside", "Stadium", 4, 0.78),
        ("Riverside", "HighSchool", 8, 0.82),
        ("Riverside", "CentralSchool", 9, 0.80),
        ("CentralSchool", "Command", 4, 0.93),
        ("HighSchool", "Command", 7, 0.91),
        ("Stadium", "Command", 8, 0.90),
    ]
    for a, b, minutes, quality in road_data:
        roads.add_road(a, b, minutes, quality)

    areas = [
        AffectedArea("A1", "North Bank", "North", 55, 5, 0.88, 0.90),
        AffectedArea("A2", "Old Market", "Market", 40, 4, 0.42, 0.78),
        AffectedArea("A3", "Riverside Colony", "Riverside", 60, 5, 0.96, 0.80),
    ]

    shelters = [
        Shelter("S1", "Central School Shelter", "CentralSchool", 70),
        Shelter("S2", "High School Shelter", "HighSchool", 100),
        Shelter("S3", "City Stadium Shelter", "Stadium", 100),
    ]

    teams = [
        RescueTeam("T1", "North Boat Team", "NorthHub", 30, 3, 3,
                   {"boat", "medical"}),
        RescueTeam("T2", "Market Bus Team", "MarketHub", 22, 3, 3,
                   {"bus"}),
        RescueTeam("T3", "South Boat Team", "SouthHub", 32, 3, 3,
                   {"boat", "medical"}),
        RescueTeam("T4", "West Support Team", "WestHub", 25, 3, 3,
                   {"bus", "medical"}),
    ]

    kb = KnowledgeBase()
    for area in areas:
        kb.add_entity(area.area_id, area)
        kb.add_relation(area.area_id, "located_at", area.node)
    for shelter in shelters:
        kb.add_entity(shelter.shelter_id, shelter)
        kb.add_relation(shelter.shelter_id, "located_at", shelter.node)
    for team in teams:
        kb.add_entity(team.team_id, team)
        kb.add_relation(team.team_id, "based_at", team.base_node)
        for equipment in sorted(team.equipment):
            kb.add_relation(team.team_id, "equipped_with", equipment)
    return roads, areas, shelters, teams, kb


@dataclass
class SimulationResult:
    scenario_name: str
    total_time: float
    people_safe: int
    total_population: int
    fuel_units_used: int
    trips_used: int
    assignments: List[Dict[str, Any]]
    shelter_occupancy: Dict[str, int]
    logs: List[str]

    @property
    def evacuation_percentage(self) -> float:
        if self.total_population == 0:
            return 0.0
        return 100.0 * self.people_safe / self.total_population


def required_equipment_for_area(area: AffectedArea,
                                fuzzy_values: Dict[str, float]) -> Set[str]:
    required: Set[str] = set()
    if fuzzy_values["flood_high"] >= 0.55:
        required.add("boat")
    return required


def run_simulation(disrupted: bool) -> SimulationResult:
    roads, areas, shelters, teams, kb = build_initial_state()
    scenario_name = "Disrupted scenario" if disrupted else "Baseline scenario"
    logs: List[str] = [f"\n========== {scenario_name} =========="]

    fuzzy_values = {
        area.area_id: estimate_fuzzy_risk(area, roads, shelters)
        for area in areas
    }
    forward_chain(kb, areas, fuzzy_values)

    priority_scores = {
        area.area_id: calculate_priority_score(area, fuzzy_values[area.area_id])
        for area in areas
    }
    ordered_areas = sorted(areas,
                           key=lambda area: priority_scores[area.area_id],
                           reverse=True)

    logs.append("Knowledge-base facts derived by forward chaining:")
    for fact in sorted(kb.facts, key=lambda item: tuple(str(x) for x in item)):
        if fact[0] in {"priority", "requires_equipment", "rescue_action",
                       "resource_action"}:
            logs.append(f"  IF-THEN result: {fact}")

    coordinator = ContractNetCoordinator(teams)
    assignments: List[Dict[str, Any]] = []
    total_time = 0.0
    people_safe = 0
    trips_used = 0
    fuel_used = 0

    for index, area in enumerate(ordered_areas):
        if disrupted and index == 1:
            # Unexpected event after the first completed rescue task.
            roads.block_road("North", "HighSchool")
            central = next(s for s in shelters if s.shelter_id == "S1")
            central.open = False
            logs.append("\n*** UNEXPECTED EVENT ***")
            logs.append("Road North <-> HighSchool is blocked.")
            logs.append("Central School Shelter becomes unavailable.")
            logs.append("The coordinator re-announces remaining tasks. Agents "
                        "submit new bids using the updated roads and shelter capacity.")

        required = required_equipment_for_area(area, fuzzy_values[area.area_id])
        winner, bids = coordinator.announce_and_award(
            area, roads, shelters, required
        )

        logs.append(f"\nTask announcement: evacuate {area.area_id} - {area.name} "
                    f"({area.population} people)")
        logs.append(f"  Priority score: {priority_scores[area.area_id]:.2f}; "
                    f"fuzzy risk: {fuzzy_values[area.area_id]['overall_risk']:.2f}")
        logs.append(f"  Required equipment: {sorted(required) if required else 'none'}")
        if bids:
            logs.append("  CNP bids (lower is better):")
            for bid in bids:
                logs.append(f"    {bid.team_id}: {bid.bid_value:.2f}")

        if winner is None:
            logs.append("  No feasible team/shelter bid. Area could not be evacuated.")
            continue

        commit_award(winner, teams, shelters)
        total_time += winner.estimated_mission_time
        people_safe += area.population
        trips_used += winner.total_trips
        fuel_used += winner.total_trips

        shelter_details = []
        for allocation in winner.shelter_allocations:
            shelter = next(s for s in shelters
                           if s.shelter_id == allocation.shelter_id)
            shelter_details.append({
                "shelter": shelter.name,
                "people": allocation.people,
                "path": allocation.path,
                "trips": allocation.trips,
            })

        assignments.append({
            "area": area.name,
            "area_id": area.area_id,
            "team_id": winner.team_id,
            "team_path": winner.base_to_area_path,
            "shelters": shelter_details,
            "mission_time": winner.estimated_mission_time,
            "trips": winner.total_trips,
            "people": area.population,
        })
        logs.append(f"  Awarded to {winner.team_id}.")
        logs.append(f"  Team route: {' -> '.join(winner.base_to_area_path)}")
        logs.append(f"  Estimated mission time: {winner.estimated_mission_time:.2f} minutes")
        for detail in shelter_details:
            logs.append(f"  Shelter: {detail['shelter']}; people: {detail['people']}; "
                        f"trips: {detail['trips']}; route: "
                        f"{' -> '.join(detail['path'])}")

    shelter_occupancy = {shelter.name: shelter.occupied for shelter in shelters}
    total_population = sum(area.population for area in areas)

    logs.append("\nScenario evaluation:")
    logs.append(f"  People evacuated safely: {people_safe}/{total_population}")
    logs.append(f"  Evacuation time (sequential mission estimate): {total_time:.2f} minutes")
    logs.append(f"  Fuel/resource units used: {fuel_used}")
    logs.append(f"  Transport trips used: {trips_used}")
    logs.append(f"  Shelter occupancy: {shelter_occupancy}")

    return SimulationResult(
        scenario_name=scenario_name,
        total_time=total_time,
        people_safe=people_safe,
        total_population=total_population,
        fuel_units_used=fuel_used,
        trips_used=trips_used,
        assignments=assignments,
        shelter_occupancy=shelter_occupancy,
        logs=logs,
    )


# -----------------------------------------------------------------------------
# VISUALIZATION AND MAIN PROGRAM
# -----------------------------------------------------------------------------

def print_knowledge_representation() -> None:
    _, areas, shelters, teams, kb = build_initial_state()
    print("\n========== KNOWLEDGE REPRESENTATION ==========")
    print("Entities are represented as Python objects; relationships are triples.")
    for relation in kb.relations:
        print("  ", relation)


def print_comparison(baseline: SimulationResult,
                     disrupted: SimulationResult) -> None:
    print("\n========== BASELINE VS DISRUPTED COMPARISON ==========")
    header = f"{'Metric':<32}{'Baseline':>16}{'Disrupted':>16}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Evacuation time (minutes)", baseline.total_time, disrupted.total_time),
        ("People evacuated safely", baseline.people_safe, disrupted.people_safe),
        ("Evacuation percentage", baseline.evacuation_percentage,
         disrupted.evacuation_percentage),
        ("Fuel/resource units", baseline.fuel_units_used,
         disrupted.fuel_units_used),
        ("Transport trips", baseline.trips_used, disrupted.trips_used),
    ]
    for name, base_value, disrupted_value in rows:
        if "percentage" in name:
            print(f"{name:<32}{base_value:>15.1f}%{disrupted_value:>15.1f}%")
        else:
            print(f"{name:<32}{base_value:>16.2f}{disrupted_value:>16.2f}")


def create_visualization(baseline: SimulationResult,
                         disrupted: SimulationResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib is not installed. Install it with: "
              "python -m pip install matplotlib")
        return

    labels = ["Baseline", "Disrupted"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].bar(labels, [baseline.total_time, disrupted.total_time])
    axes[0].set_title("Estimated evacuation time")
    axes[0].set_ylabel("Minutes")

    axes[1].bar(labels, [baseline.fuel_units_used, disrupted.fuel_units_used])
    axes[1].set_title("Resource usage")
    axes[1].set_ylabel("Fuel / resource units")

    axes[2].bar(labels, [baseline.people_safe, disrupted.people_safe])
    axes[2].set_title("People evacuated safely")
    axes[2].set_ylabel("People")

    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)

    fig.suptitle("Project LAST EXIT - Scenario Comparison", fontsize=14)
    fig.tight_layout()
    output_file = Path(__file__).resolve().parent / "lab11_u23ai093_results.png"
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"\nVisualization saved as: {output_file}")
    # Avoid a GUI warning when running in a headless environment.
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main() -> None:
    print("PROJECT LAST EXIT - INTELLIGENT DISASTER EVACUATION SYSTEM")
    print_knowledge_representation()

    baseline = run_simulation(disrupted=False)
    disrupted = run_simulation(disrupted=True)

    for line in baseline.logs:
        print(line)
    for line in disrupted.logs:
        print(line)

    print_comparison(baseline, disrupted)
    create_visualization(baseline, disrupted)

    total_population = baseline.total_population
    print("\n========== OBSERVATION ==========")
    print("The system represents teams, affected areas, and shelters as entities "
          "connected by relationships. Forward-chaining rules infer priorities "
          "and rescue actions, while fuzzy membership values handle uncertain "
          "flood levels, report reliability, and road quality.")
    print("The Contract Net Protocol lets eligible rescue-team agents submit bids. "
          "The coordinator awards each task to the feasible team with the lowest "
          "bid and reserves fuel and shelter capacity.")
    print("In the disrupted scenario, a road is blocked and a shelter is closed "
          "after the first rescue. Remaining tasks are re-announced and agents "
          "calculate new routes and bids using the updated network.")
    if disrupted.people_safe < total_population:
        print("Some people could not be evacuated because the remaining teams, "
              "routes, resources, or shelter capacity were insufficient.")
    else:
        print("All simulated people are assigned to available shelters in both "
              "scenarios. Compare the printed time and resource metrics to see "
              "the impact of the unexpected event.")
    print("\nNote: This is a simplified educational simulation, not a real-world "
          "emergency dispatch system.")


if __name__ == "__main__":
    main()
