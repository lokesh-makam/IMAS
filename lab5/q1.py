
# Experiment 4: Backward Chaining Engine with Askable Facts
# Application: Car Fault Diagnosis

# Questions that the engine can ask the user
FACT_QUESTIONS = {
    "headlights_dim": "Are the headlights dim?",
    "clicking_sound": "Do you hear a clicking sound when turning the key?",
    "engine_not_cranking": "Does the engine fail to crank (does not turn over)?",
    "fuel_gauge_empty": "Does the fuel gauge show that the tank is empty?",
    "engine_cranks_but_wont_start":
        "Does the engine crank but fail to start?",
    "fuel_available": "Is there fuel in the tank?"
}


# Rules represent: IF all premises are true, THEN the fault is identified.
# The order of these rules is the hypothesis order.
RULES = [
    {
        "fault": "Dead battery",
        "premises": ("headlights_dim", "clicking_sound"),
        "reason": "A weak or dead battery can cause dim headlights "
                  "and a clicking sound when starting the car."
    },
    {
        "fault": "Starter motor fault",
        "premises": ("clicking_sound", "engine_not_cranking"),
        "reason": "A faulty starter motor can produce a clicking sound "
                  "while the engine fails to crank."
    },
    {
        "fault": "Empty fuel tank",
        "premises": ("fuel_gauge_empty",),
        "reason": "An empty fuel tank can prevent the car from starting."
    },
    {
        "fault": "Ignition system fault",
        "premises": ("engine_cranks_but_wont_start", "fuel_available"),
        "reason": "If the engine cranks, fails to start, and fuel is "
                  "available, an ignition or related engine fault may exist."
    }
]


class DiagnosisStopped(Exception):
    """Used when the user chooses to stop the diagnosis."""
    pass


def show_why(fact, rule):
    """Explain why the current question is being asked."""

    conditions = " AND ".join(
        FACT_QUESTIONS[premise] for premise in rule["premises"]
    )

    print("\n--- WHY is this question being asked? ---")
    print(f"Current hypothesis: {rule['fault']}")
    print(f"Rule: IF {conditions},")
    print(f"      THEN {rule['fault']}.")
    print(f"Current condition: {FACT_QUESTIONS[fact]}")
    print(f"Reason: {rule['reason']}")
    print("The engine needs to verify the rule's conditions "
          "before accepting this hypothesis.")
    print("-----------------------------------------")


def ask_fact(fact, rule, known_facts, asked_facts):
    """Ask an unknown fact and remember the user's answer."""

    # Reuse an answer if this fact has already been checked.
    if fact in known_facts:
        answer = "YES" if known_facts[fact] else "NO"
        print(f"  Previously answered: {FACT_QUESTIONS[fact]} -> {answer}")
        return known_facts[fact]

    if fact not in asked_facts:
        asked_facts.append(fact)

    while True:
        print(f"\nQUESTION: {FACT_QUESTIONS[fact]}")
        response = input("Enter yes / no / why / quit: ").strip().lower()

        if response in ("yes", "y"):
            known_facts[fact] = True
            return True

        elif response in ("no", "n"):
            known_facts[fact] = False
            return False

        elif response in ("why", "w"):
            show_why(fact, rule)
            # Ask the same question again.
            continue

        elif response in ("quit", "q"):
            raise DiagnosisStopped

        else:
            print("Invalid input. Please enter yes, no, why, or quit.")


def display_facts(asked_facts, known_facts):
    """Display the facts asked during the diagnosis."""

    print("\n--- Facts Asked During Diagnosis ---")

    if not asked_facts:
        print("No facts were asked.")
        return

    for fact in asked_facts:
        answer = known_facts.get(fact)

        if answer is None:
            result = "NOT ANSWERED"
        elif answer:
            result = "YES"
        else:
            result = "NO"

        print(f"- {FACT_QUESTIONS[fact]} -> {result}")


def backward_chaining():
    """Try to prove each fault hypothesis in order."""

    # Answers are reset for each new diagnosis.
    known_facts = {}
    asked_facts = []
    diagnosis_found = False

    print("=" * 50)
    print("     CAR FAULT DIAGNOSIS SYSTEM")
    print("     BACKWARD CHAINING ENGINE")
    print("=" * 50)

    print("\nHypotheses are checked in this order:")
    for number, rule in enumerate(RULES, start=1):
        print(f"{number}. {rule['fault']}")

    print("\nAnswer the questions using yes, no, why, or quit.")

    try:
        for rule in RULES:
            print(f"\nChecking hypothesis: {rule['fault']}")

            rule_satisfied = True

            # Every premise must be true for a rule to succeed.
            for fact in rule["premises"]:
                answer = ask_fact(
                    fact, rule, known_facts, asked_facts
                )

                if not answer:
                    rule_satisfied = False
                    print(f"Hypothesis rejected: {rule['fault']}")
                    break

            # Stop at the first hypothesis that is proved.
            if rule_satisfied:
                diagnosis_found = True
                print("\n" + "=" * 50)
                print(f"DIAGNOSIS: {rule['fault']}")
                print(f"Reason: {rule['reason']}")
                print("=" * 50)
                break

        if not diagnosis_found:
            print("\nDIAGNOSIS: No matching fault found.")
            print("The current rules could not identify a fault.")

    except DiagnosisStopped:
        print("\nDiagnosis stopped by the user.")

    display_facts(asked_facts, known_facts)


if __name__ == "__main__":
    backward_chaining()
