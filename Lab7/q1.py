
import itertools
import matplotlib.pyplot as plt

# ==================================================
# EXERCISE 1: PROPOSITIONAL LOGIC
# If it rains, then the road is wet.
# ==================================================

print("\n1. PROPOSITIONAL LOGIC")
print("P = It rains, Q = Road is wet")
print(f"{'P':<12}{'Q':<12}{'P -> Q':<12}")

for P, Q in itertools.product([False, True], repeat=2):
    # P -> Q is equivalent to (NOT P) OR Q
    result = (not P) or Q
    print(f"{str(P):<12}{str(Q):<12}{str(result):<12}")


# ==================================================
# EXERCISE 2: FIRST-ORDER LOGIC
# All humans are mortal. Socrates is human.
# Therefore, Socrates is mortal.
# ==================================================

print("\n2. FIRST-ORDER LOGIC")

# Facts: Socrates is a human.
humans = {"Socrates"}

# Rule: All humans are mortal.
def is_mortal(person):
    return person in humans

person = "Socrates"

is_human = person in humans
mortal = is_mortal(person) if is_human else False

print(f"Human({person}) = {is_human}")
print(f"Mortal({person}) = {mortal}")


# ==================================================
# EXERCISE 3: PROBABILITY OF EVENTS
# A = Passed, B = Attended
# ==================================================

print("\n3. PROBABILITY OF EVENTS")

# Each tuple contains (attended, passed).
students = [
    (True, True),
    (True, True),
    (True, True),
    (True, False),
    (True, True),
    (True, False),
    (False, True),
    (False, False),
    (False, False),
    (False, True)
]

total = len(students)

# Count individual events.
passed = sum(p for a, p in students)
failed = total - passed

attended = sum(a for a, p in students)
absent = total - attended

# Count joint and union events.
both = sum(a and p for a, p in students)
either = sum(a or p for a, p in students)

# Calculate probabilities.
P_A = passed / total
P_not_A = failed / total
P_B = attended / total
P_not_B = absent / total
P_both = both / total
P_either = either / total

print(f"P(Passed) = {P_A:.2f}")
print(f"P(Failed) = {P_not_A:.2f}")
print(f"P(Attended) = {P_B:.2f}")
print(f"P(Not Attended) = {P_not_B:.2f}")
print(f"P(Passed AND Attended) = {P_both:.2f}")
print(f"P(Passed OR Attended) = {P_either:.2f}")


# ==================================================
# EXERCISE 4: CONDITIONAL PROBABILITY
# ==================================================

print("\n4. CONDITIONAL PROBABILITY")

# P(A | B) = P(A AND B) / P(B)
P_A_given_B = both / attended if attended > 0 else 0

# P(B | A) = P(A AND B) / P(A)
P_B_given_A = both / passed if passed > 0 else 0

print(f"P(Passed | Attended) = {P_A_given_B:.2f}")
print(f"P(Attended | Passed) = {P_B_given_A:.2f}")


# ==================================================
# EXERCISE 5: BAYES' THEOREM
# Disease diagnosis after a positive test.
# ==================================================

print("\n5. BAYES' THEOREM")

sensitivity = 0.99
false_positive_rate = 0.05


def bayes_theorem(prior, sensitivity, false_positive_rate):
    # P(Positive AND Disease)
    numerator = sensitivity * prior

    # Total probability of a positive test
    denominator = (
        sensitivity * prior
        + false_positive_rate * (1 - prior)
    )

    if denominator == 0:
        return 0

    return numerator / denominator


# Test different disease prevalence values.
prevalence_values = [0.01, 0.02, 0.05]
bayes_results = []

print("\nProbability of disease after a positive test:")

for prevalence in prevalence_values:
    result = bayes_theorem(
        prevalence,
        sensitivity,
        false_positive_rate
    )

    bayes_results.append(result)

    print(
        f"Prevalence = {prevalence:.0%} "
        f"-> P(Disease | Positive) = {result:.2%}"
    )


# ==================================================
# EXERCISE 6: VISUALIZATION USING MATPLOTLIB
# ==================================================

print("\n6. VISUALIZATION")

# Graph 1: Student probabilities.
labels = [
    "Passed",
    "Failed",
    "Attended",
    "Absent",
    "Passed AND Attended",
    "Passed OR Attended",
    "Passed | Attended",
    "Attended | Passed"
]

student_probabilities = [
    P_A,
    P_not_A,
    P_B,
    P_not_B,
    P_both,
    P_either,
    P_A_given_B,
    P_B_given_A
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].bar(
    labels,
    [p * 100 for p in student_probabilities]
)

axes[0].set_title("Student Event Probabilities")
axes[0].set_ylabel("Probability (%)")
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis="x", rotation=55)
axes[0].grid(axis="y", alpha=0.3)


# Graph 2: Bayes' Theorem.
prevalence_labels = [
    f"Prevalence {p:.0%}"
    for p in prevalence_values
]

posterior_percentages = [
    result * 100 for result in bayes_results
]

bars = axes[1].bar(
    prevalence_labels,
    posterior_percentages
)

axes[1].set_title("Bayes' Theorem: Effect of Prevalence")
axes[1].set_ylabel(
    "Probability of Disease After Positive Test (%)"
)
axes[1].set_ylim(
    0,
    max(posterior_percentages) * 1.25
)
axes[1].grid(axis="y", alpha=0.3)

# Display values above each bar.
for bar, value in zip(bars, posterior_percentages):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

print("\nExperiment 7 completed.")
