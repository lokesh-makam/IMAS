"""
IMAES Lab 2: Production Rules and the Match Phase
U23AI055
"""
import time
import random

# Core Engine + Exercise 2
class Rule:
    def __init__(self, rule_id, conditions, conclusion):
        # Encode each rule as a record with an ID, conditions & conclusion
        self.id = rule_id
        self.conditions = conditions
        self.conclusion = conclusion

    def __str__(self):
        return f"Rule {self.id}: IF {' AND '.join(self.conditions)} THEN {self.conclusion}"

class ProductionSystem:
    def __init__(self):
        self.rules = []
        self.wm = set() # working memory as a set of facts

    def add_rule(self, rule):
        self.rules.append(rule)

    def match(self):
        conflict_set = []

        for rule in self.rules:
            # Refractoriness Check: Is conclusion already in WM?
            if rule.conclusion in self.wm:
                continue

            conditions_met = True
            for cond in rule.conditions:
                # Negated Conditions ('NOT fact')
                if cond.startswith("NOT "):
                    positive_fact = cond[4:]
                    if positive_fact in self.wm:
                        conditions_met = False
                        break
                # Standard condition check
                else:
                    if cond not in self.wm:
                        conditions_met = False
                        break

            # If all conditions are met, rule is applicable
            if conditions_met:
                conflict_set.append(rule)

        return conflict_set

# Excercise 1 : Dynamic Working Memory
def run_exercise_1_and_2():
    print("==========================================================")
    print("EXERCISE 1 & 2: DYNAMIC WM AND NEGATED CONDITIONS")
    print("==========================================================\n")

    ps = ProductionSystem()

    ps.add_rule(Rule('R1', ["has_feathers", "lays_eggs"], "bird"))
    ps.add_rule(Rule("R2", ["bird", "cannot_fly"], "penguin"))
    ps.add_rule(Rule("R3", ["bird", "NOT cannot_fly"], "can_fly"))
    ps.add_rule(Rule("R4", ["penguin", "can_swim"], "swimming_penguin"))

    print("--- Rule Base ---")
    for r in ps.rules:
        print(r)
    print("-----------------\n")

    facts_to_add = ["has_feathers", "lays_eggs", "bird", "cannot_fly", "can_swim"]

    for fact in facts_to_add:
        print(f"-> Adding fact to WM: '{fact}'")
        ps.wm.add(fact)

        conflict_set = ps.match()

        print(f"Current WM  : {ps.wm}")
        if conflict_set:
            print(f"Conflict Set    : {[r.id for r in conflict_set]}\n")
        else:
            print("Conflict Set     : [] (Empty)\n")

# Exercise 3 : Performance Benchmarking
def run_exercise_3():
    print("==========================================================")
    print("EXERCISE 3: MATCH PHASE PERFORMANCE BENCHMARKING")
    print("==========================================================\n")

    ps_perf = ProductionSystem()
    num_rules = 1000

    # 1. Generate 1,000 auto-generated rules
    print(f"Generating {num_rules} dummy rules...")
    for i in range(num_rules):
        ps_perf.add_rule(Rule(
            rule_id=f"Dummy_R{i}", 
            conditions=[f"fact_A_{i}", f"fact_B_{i}"], 
            conclusion=f"conclusion_{i}"
        ))

    # 2. Populate Working Memory so about half the rules will match
    print("Populating Working Memory...")
    for i in range(num_rules // 2):
        ps_perf.wm.add(f"fact_A_{i}")
        ps_perf.wm.add(f"fact_B_{i}")

    # 3. Measure match time using time.perf_counter
    print("Executing Match Phase...")
    start_time = time.perf_counter()
    conflict_set = ps_perf.match()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nResults:")
    print(f"- Rules Evaluated : {len(ps_perf.rules)}")
    print(f"- Rules Matched   : {len(conflict_set)}")
    print(f"- Match Time      : {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    run_exercise_1_and_2()
    run_exercise_3()