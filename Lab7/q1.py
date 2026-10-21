
# Experiment 7: Logical and Probabilistic Reasoning
# Propositional Logic, First-Order Logic, Probability, Bayes' Theorem

import matplotlib.pyplot as plt
from itertools import product


# ============================================================
# EXERCISE 1: PROPOSITIONAL LOGIC
# Rule: If it rains, then the road is wet.
# ============================================================

def propositional_logic():
    print("\n" + "=" * 60)
    print("EXERCISE 1: PROPOSITIONAL LOGIC")
    print("=" * 60)

    print("P = It rains")
    print("Q = The road is wet")
    print("Rule: P -> Q (If P, then Q)\n")

    print(f"{'P (Rains)':<15}{'Q (Road Wet)':<18}{'P -> Q':<12}")
    print("-" * 45)

    results = []

    for rains, road_wet in product([False, True], repeat=2):
        # P -> Q is equivalent to NOT P OR Q.
        implication = (not rains) or road_wet

        results.append((rains, road_wet, implication))

        print(
            f"{str(rains):<15}{str(road_wet):<18}"
            f"{str(implication):<12}"
        )

    violations = [
        (rains, road_wet)
        for rains, road_wet, implication in results
        if not implication
    ]

    print("\nVerification:")
    if violations:
        print("The rule is FALSE only when it rains but the road is not wet.")
        print("This agrees with the truth table.")
    else:
        print("The rule is TRUE for all combinations.")


# ============================================================
# EXERCISE 2: FIRST-ORDER LOGIC
# All humans are mortal. Socrates is a human.
# Therefore, Socrates is mortal.
# ============================================================

def first_order_logic():
    print("\n" + "=" * 60)
    print("EXERCISE 2: FIRST-ORDER LOGIC")
    print("=" * 60)

    print("Rule: For every x, if Human(x), then Mortal(x).")
    print("Fact: Human(Socrates).")

    humans = {"Socrates"}

    def is_mortal(person):
        # Apply the rule: every human is mortal.
        return person in humans

    person = "Socrates"

    if person in humans:
        print(f"\nHuman({person}) = True")
    else:
        print(f"\nHuman({person}) = False")

    if is_mortal(person):
        print(f"Mortal({person}) = True")
        print("Conclusion: Socrates is mortal.")
    else:
        print(f"Mortal({person}) = False")
        print("Conclusion: Mortality cannot be derived.")


# ============================================================
# EXERCISE 3: PROBABILITY OF EVENTS
# A = student passed
# B = student attended classes
# ============================================================

def create_dataset():
    return [
        {"student": "S1", "attended": True,  "passed": True},
        {"student": "S2", "attended": True,  "passed": True},
        {"student": "S3", "attended": True,  "passed": True},
        {"student": "S4", "attended": True,  "passed": False},
        {"student": "S5", "attended": True,  "passed": True},
        {"student": "S6", "attended": True,  "passed": False},
        {"student": "S7", "attended": False, "passed": True},
        {"student": "S8", "attended": False, "passed": False},
        {"student": "S9", "attended": False, "passed": False},
        {"student": "S10", "attended": False, "passed": True},
    ]


def probability_analysis(dataset):
    print("\n" + "=" * 60)
    print("EXERCISE 3: PROBABILITY OF EVENTS")
    print("=" * 60)

    total = len(dataset)

    count_A = sum(student["passed"] for student in dataset)
    count_B = sum(student["attended"] for student in dataset)

    count_intersection = sum(
        student["passed"] and student["attended"]
        for student in dataset
    )

    count_union = sum(
        student["passed"] or student["attended"]
        for student in dataset
    )

    # Individual probabilities
    p_A = count_A / total
    p_B = count_B / total

    # Joint probability
    p_intersection = count_intersection / total

    # Probability of union
    p_union = count_union / total

    print(f"Total students: {total}")
    print(f"Students who passed: {count_A}")
    print(f"Students who attended: {count_B}")
    print(f"Students who both attended and passed: {count_intersection}")
    print(f"Students who passed or attended: {count_union}")

    print("\nCalculated probabilities:")
    print(f"P(A)       = {p_A:.3f} ({p_A * 100:.1f}%)")
    print(f"P(B)       = {p_B:.3f} ({p_B * 100:.1f}%)")
    print(f"P(A and B) = {p_intersection:.3f} "
          f"({p_intersection * 100:.1f}%)")
    print(f"P(A or B)  = {p_union:.3f} ({p_union * 100:.1f}%)")

    return {
        "P(A)": p_A,
        "P(B)": p_B,
        "P(A and B)": p_intersection,
        "P(A or B)": p_union,
        "count_A": count_A,
        "count_B": count_B,
        "count_intersection": count_intersection,
        "total": total,
    }


# ============================================================
# EXERCISE 4: CONDITIONAL PROBABILITY
# P(A|B) = probability of passing given attendance
# P(B|A) = probability of attendance given passing
# ============================================================

def conditional_probability(probabilities):
    print("\n" + "=" * 60)
    print("EXERCISE 4: CONDITIONAL PROBABILITY")
    print("=" * 60)

    count_A = probabilities["count_A"]
    count_B = probabilities["count_B"]
    count_intersection = probabilities["count_intersection"]

    # P(A|B) = P(A and B) / P(B)
    p_A_given_B = (
        count_intersection / count_B if count_B > 0 else None
    )

    # P(B|A) = P(A and B) / P(A)
    p_B_given_A = (
        count_intersection / count_A if count_A > 0 else None
    )

    if p_A_given_B is not None:
        print(
            f"P(A|B) = P(Passed | Attended) = "
            f"{p_A_given_B:.3f} ({p_A_given_B * 100:.1f}%)"
        )
    else:
        print("P(A|B) is undefined because no students attended.")

    if p_B_given_A is not None:
        print(
            f"P(B|A) = P(Attended | Passed) = "
            f"{p_B_given_A:.3f} ({p_B_given_A * 100:.1f}%)"
        )
    else:
        print("P(B|A) is undefined because no students passed.")

    return p_A_given_B, p_B_given_A


# ============================================================
# EXERCISE 5: BAYES' THEOREM
# Estimate disease probability after a positive test.
# ============================================================

def bayes_theorem(prevalence, sensitivity, false_positive_rate):
    """
    prevalence: P(Disease)
    sensitivity: P(Positive | Disease)
    false_positive_rate: P(Positive | No Disease)
    """

    true_positive_probability = prevalence * sensitivity

    positive_test_probability = (
        prevalence * sensitivity
        + (1 - prevalence) * false_positive_rate
    )

    if positive_test_probability == 0:
        return None

    posterior = (
        true_positive_probability / positive_test_probability
    )

    return posterior


def read_modified_prevalence():
    print("\nChange disease prevalence for the graph.")
    print("Press Enter to use the example value of 2%.")

    while True:
        value = input(
            "Enter modified disease prevalence in percent (0-100): "
        ).strip()

        if value == "":
            return 0.02

        try:
            percentage = float(value)

            if 0 <= percentage <= 100:
                return percentage / 100

            print("Enter a percentage between 0 and 100.")

        except ValueError:
            print("Invalid input. Enter a number or press Enter.")


def run_bayes_example():
    print("\n" + "=" * 60)
    print("EXERCISE 5: BAYES' THEOREM")
    print("=" * 60)

    # Initial probabilities
    prior = 0.01
    sensitivity = 0.99
    false_positive_rate = 0.05

    baseline_posterior = bayes_theorem(
        prior, sensitivity, false_positive_rate
    )

    positive_probability = (
        prior * sensitivity
        + (1 - prior) * false_positive_rate
    )

    print(f"Prior disease probability: {prior * 100:.1f}%")
    print(f"Test sensitivity: {sensitivity * 100:.1f}%")
    print(f"False positive rate: {false_positive_rate * 100:.1f}%")
    print(f"Probability of a positive test: "
          f"{positive_probability * 100:.2f}%")
    print(f"Probability of disease after a positive test: "
          f"{baseline_posterior * 100:.2f}%")

    modified_prior = read_modified_prevalence()

    modified_posterior = bayes_theorem(
        modified_prior, sensitivity, false_positive_rate
    )

    print("\nAfter changing the disease prevalence:")
    print(f"Modified prior: {modified_prior * 100:.1f}%")
    print(f"New probability of disease after a positive test: "
          f"{modified_posterior * 100:.2f}%")

    return (
        baseline_posterior,
        modified_posterior,
        modified_prior,
    )


# ============================================================
# EXERCISE 6: VISUALIZATION USING MATPLOTLIB
# ============================================================

def visualize_probabilities(
    probabilities,
    conditional_values,
    bayes_results,
):
    print("\n" + "=" * 60)
    print("EXERCISE 6: VISUALIZATION")
    print("=" * 60)

    p_A_given_B, p_B_given_A = conditional_values
    baseline_posterior, modified_posterior, modified_prior = bayes_results

    labels = [
        "P(A)",
        "P(B)",
        "P(A and B)",
        "P(A or B)",
        "P(A|B)",
        "P(B|A)",
    ]

    values = [
        probabilities["P(A)"],
        probabilities["P(B)"],
        probabilities["P(A and B)"],
        probabilities["P(A or B)"],
        p_A_given_B,
        p_B_given_A,
    ]

    # Convert probabilities into percentages for display.
    percentages = [value * 100 for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Graph 1: Student dataset probabilities
    bars1 = axes[0].bar(labels, percentages)
    axes[0].set_title("Student Event Probabilities")
    axes[0].set_ylabel("Probability (%)")
    axes[0].set_ylim(0, 110)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars1, percentages):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    # Graph 2: Effect of changing the disease prevalence
    bayes_labels = [
        "Original\nprevalence 1%",
        f"Modified\nprevalence {modified_prior * 100:.1f}%",
    ]

    bayes_percentages = [
        baseline_posterior * 100,
        modified_posterior * 100,
    ]

    bars2 = axes[1].bar(bayes_labels, bayes_percentages)
    axes[1].set_title("Disease Probability After Positive Test")
    axes[1].set_ylabel("Posterior probability (%)")
    axes[1].set_ylim(
        0, max(10, max(bayes_percentages) * 1.25)
    )
    axes[1].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars2, bayes_percentages):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.suptitle("Experiment 7: Logical and Probabilistic Reasoning")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM: RUN ALL SIX EXERCISES
# ============================================================

def main():
    propositional_logic()

    first_order_logic()

    dataset = create_dataset()
    probabilities = probability_analysis(dataset)

    conditional_values = conditional_probability(probabilities)

    bayes_results = run_bayes_example()

    visualize_probabilities(
        probabilities,
        conditional_values,
        bayes_results,
    )

    print("\nExperiment 7 completed successfully.")


if __name__ == "__main__":
    main()
