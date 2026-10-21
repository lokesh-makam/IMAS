
# Experiment 6: Fuzzy Logic and Constraint Satisfaction
# Application: Student Performance and Timetable Generation


# ============================================================
# PART 1: FUZZY LOGIC FOR STUDENT PERFORMANCE
# ============================================================

# Membership functions for attendance and marks.
# Each tuple represents (a, b, c) of a triangular membership function.
# a == b creates a left shoulder; b == c creates a right shoulder.

ATTENDANCE_SETS = {
    "low": (0, 0, 50),
    "medium": (25, 50, 75),
    "high": (50, 100, 100)
}

MARKS_SETS = {
    "low": (0, 0, 50),
    "medium": (25, 50, 75),
    "high": (50, 100, 100)
}

# Output membership functions for student performance.
PERFORMANCE_SETS = {
    "poor": (0, 0, 40),
    "average": (25, 50, 75),
    "good": (55, 75, 95),
    "excellent": (80, 100, 100)
}

# Rules: IF attendance is X AND marks are Y, THEN performance is Z.
# Exercise 3: Modify these rules to observe changes in the score.

FUZZY_RULES = [
    ("low", "low", "poor"),
    ("low", "medium", "poor"),
    ("low", "high", "average"),

    ("medium", "low", "poor"),
    ("medium", "medium", "average"),
    ("medium", "high", "good"),

    ("high", "low", "average"),
    ("high", "medium", "good"),
    ("high", "high", "excellent")
]


def triangular_membership(x, a, b, c):
    """Calculate the membership degree of x in a fuzzy set."""

    # Left shoulder
    if a == b and x <= b:
        return 1.0

    # Right shoulder
    if b == c and x >= b:
        return 1.0

    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a)

    return (c - x) / (c - b)


def calculate_memberships(value, fuzzy_sets):
    """Calculate membership degrees for all labels."""

    return {
        label: triangular_membership(value, *parameters)
        for label, parameters in fuzzy_sets.items()
    }


def performance_label(score):
    """Convert the numeric performance score into a readable label."""

    if score < 35:
        return "Poor"
    elif score < 60:
        return "Average"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"


def predict_performance(attendance, marks):
    """Apply fuzzy rules and centroid defuzzification."""

    attendance_memberships = calculate_memberships(
        attendance, ATTENDANCE_SETS
    )

    marks_memberships = calculate_memberships(
        marks, MARKS_SETS
    )

    # Output universe: every integer score from 0 to 100.
    output_values = list(range(101))
    aggregated_output = [0.0] * len(output_values)
    activated_rules = []

    # Evaluate each rule.
    for attendance_label, marks_label, result_label in FUZZY_RULES:

        # AND operator: minimum of the two membership degrees.
        strength = min(
            attendance_memberships[attendance_label],
            marks_memberships[marks_label]
        )

        if strength > 0:
            activated_rules.append(
                (attendance_label, marks_label, result_label, strength)
            )

            # Clip the output membership function using rule strength.
            for i, output_value in enumerate(output_values):
                membership = triangular_membership(
                    output_value,
                    *PERFORMANCE_SETS[result_label]
                )

                clipped_value = min(strength, membership)

                # Aggregate all rules using the maximum operator.
                aggregated_output[i] = max(
                    aggregated_output[i], clipped_value
                )

    # Centroid defuzzification: convert fuzzy output into a crisp score.
    total_membership = sum(aggregated_output)

    if total_membership == 0:
        return None, attendance_memberships, marks_memberships, activated_rules

    score = sum(
        value * degree
        for value, degree in zip(output_values, aggregated_output)
    ) / total_membership

    return score, attendance_memberships, marks_memberships, activated_rules


def get_percentage(prompt):
    """Read and validate a value between 0 and 100."""

    while True:
        try:
            value = float(input(prompt))

            if 0 <= value <= 100:
                return value

            print("Enter a value between 0 and 100.")

        except ValueError:
            print("Invalid input. Enter a numeric value.")


def run_fuzzy_system():
    """Run the student performance prediction system."""

    print("\n========== FUZZY LOGIC: STUDENT PERFORMANCE ==========")

    attendance = get_percentage("Enter attendance percentage (0-100): ")
    marks = get_percentage("Enter marks percentage (0-100): ")

    score, attendance_values, marks_values, activated_rules = (
        predict_performance(attendance, marks)
    )

    print("\n--- Attendance Membership Degrees ---")
    for label, degree in attendance_values.items():
        print(f"{label.title():10}: {degree:.2f}")

    print("\n--- Marks Membership Degrees ---")
    for label, degree in marks_values.items():
        print(f"{label.title():10}: {degree:.2f}")

    print("\n--- Activated Fuzzy Rules ---")
    for attendance_label, marks_label, result_label, strength in activated_rules:
        print(
            f"IF attendance is {attendance_label} AND "
            f"marks are {marks_label} THEN performance is "
            f"{result_label} (strength = {strength:.2f})"
        )

    if score is not None:
        print("\n--- Final Result ---")
        print(f"Attendance: {attendance:.1f}%")
        print(f"Marks:      {marks:.1f}%")
        print(f"Performance score: {score:.2f}/100")
        print(f"Performance category: {performance_label(score)}")

    print("\nNote: This is an illustrative fuzzy rule system.")


# ============================================================
# PART 2: CONSTRAINT SATISFACTION PROBLEM (TIMETABLE)
# ============================================================

# Each subject has a teacher and a student group.
# Subjects sharing a teacher or student group cannot overlap.

SUBJECT_INFO = {
    "AI": {
        "teacher": "Prof_A",
        "group": "IT-A"
    },
    "DBMS": {
        "teacher": "Prof_B",
        "group": "IT-A"
    },
    "Machine Learning": {
        "teacher": "Prof_D",
        "group": "IT-A"
    },
    "Operating Systems": {
        "teacher": "Prof_A",
        "group": "IT-B"
    },
    "Computer Networks": {
        "teacher": "Prof_C",
        "group": "IT-B"
    }
}

TIME_SLOTS = [
    "Monday 9:00-10:00",
    "Monday 10:00-11:00",
    "Tuesday 9:00-10:00",
    "Tuesday 10:00-11:00"
]

# Exercise 3: Add pairs of subjects that must never overlap.
# Example:
# EXTRA_CONFLICTS = {
#     frozenset(("AI", "Computer Networks"))
# }
#
# Keep the set empty for the original problem.

EXTRA_CONFLICTS = set()


def is_consistent(subject, slot, assignment):
    """Check whether assigning this subject to this slot is valid."""

    for other_subject, other_slot in assignment.items():

        # No conflict if the subjects have different time slots.
        if slot != other_slot:
            continue

        same_teacher = (
            SUBJECT_INFO[subject]["teacher"]
            == SUBJECT_INFO[other_subject]["teacher"]
        )

        same_group = (
            SUBJECT_INFO[subject]["group"]
            == SUBJECT_INFO[other_subject]["group"]
        )

        explicitly_forbidden = (
            frozenset((subject, other_subject)) in EXTRA_CONFLICTS
        )

        if same_teacher or same_group or explicitly_forbidden:
            return False

    return True


def backtracking_search(assignment):
    """Find a timetable using recursive backtracking."""

    # All subjects have been assigned a valid time.
    if len(assignment) == len(SUBJECT_INFO):
        return assignment.copy()

    # Select the next subject that has not been scheduled.
    subject = next(
        item for item in SUBJECT_INFO if item not in assignment
    )

    # Try every possible time slot.
    for slot in TIME_SLOTS:

        if is_consistent(subject, slot, assignment):
            assignment[subject] = slot

            result = backtracking_search(assignment)

            if result is not None:
                return result

            # Undo the assignment if it leads to no solution.
            del assignment[subject]

    # No valid time slot was found for this path.
    return None


def validate_timetable(timetable):
    """Verify that the generated timetable has no conflicts."""

    subjects = list(timetable.keys())

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            subject_a = subjects[i]
            subject_b = subjects[j]

            if timetable[subject_a] != timetable[subject_b]:
                continue

            same_teacher = (
                SUBJECT_INFO[subject_a]["teacher"]
                == SUBJECT_INFO[subject_b]["teacher"]
            )

            same_group = (
                SUBJECT_INFO[subject_a]["group"]
                == SUBJECT_INFO[subject_b]["group"]
            )

            explicitly_forbidden = (
                frozenset((subject_a, subject_b)) in EXTRA_CONFLICTS
            )

            if same_teacher or same_group or explicitly_forbidden:
                return False

    return True


def run_timetable_system():
    """Generate and display a valid timetable."""

    print("\n========== CSP: TIMETABLE GENERATION ==========")

    print("\nConstraints:")
    print("1. A teacher cannot teach two classes at the same time.")
    print("2. A student group cannot attend two classes at the same time.")
    print("3. Explicitly forbidden subject pairs cannot overlap.")

    timetable = backtracking_search({})

    if timetable is None:
        print("\nNo valid timetable could be generated.")
        print("Try adding more time slots or relaxing constraints.")
        return

    print("\n--- Generated Timetable ---")

    for subject, slot in timetable.items():
        teacher = SUBJECT_INFO[subject]["teacher"]
        group = SUBJECT_INFO[subject]["group"]

        print(
            f"{subject:22} | {slot:25} | "
            f"Teacher: {teacher:8} | Group: {group}"
        )

    if validate_timetable(timetable):
        print("\nValidation: No constraint violations found.")
    else:
        print("\nValidation: CONFLICT FOUND!")


# ============================================================
# MAIN MENU
# ============================================================

def main():
    while True:
        print("\n==========================================")
        print("           EXPERIMENT 6")
        print("1. Student performance using fuzzy logic")
        print("2. Timetable using constraint satisfaction")
        print("3. Run both systems")
        print("4. Exit")
        print("==========================================")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            run_fuzzy_system()

        elif choice == "2":
            run_timetable_system()

        elif choice == "3":
            run_fuzzy_system()
            run_timetable_system()

        elif choice == "4":
            print("Experiment completed.")
            break

        else:
            print("Invalid choice. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
