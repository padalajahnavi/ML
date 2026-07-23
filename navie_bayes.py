"""
Naive Bayes Classifier - Works with ANY CSV Dataset

Assumptions about the CSV:
1. First row = Header (attribute names)
2. Last column = Class label (target)
3. All other columns = Categorical attributes

Program Flow:
1. Ask for CSV file path
2. Read the dataset automatically
3. Ask the user for attribute values
4. Calculate probabilities step by step
5. Display the final prediction
"""

import csv
from collections import defaultdict


def load_csv(path):
    """Load CSV file and return header and rows."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]  # Skip blank rows
    return header, rows


def get_new_tuple_interactively(header, rows):
    """Ask the user for values of each attribute."""
    attr_names = header[:-1]
    new_tuple = {}

    print("\nEnter the values for the tuple you want to classify.")
    print("-" * 60)

    for i, attr in enumerate(attr_names):
        options = sorted(set(row[i] for row in rows))
        print(f"{attr} - Possible values: {options}")
        value = input(f"Enter value for '{attr}': ").strip()
        new_tuple[attr] = value

    return new_tuple


def naive_bayes(header, rows, new_tuple):
    attr_names = header[:-1]
    class_col = header[-1]
    n_total = len(rows)

    print("\n" + "=" * 60)
    print(f"Attributes   : {attr_names}")
    print(f"Class Column : {class_col}")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: Group rows by class
    # --------------------------------------------------
    classes = defaultdict(list)

    for row in rows:
        classes[row[-1]].append(row)

    print("STEP 1: Class Counts")
    for label, class_rows in classes.items():
        print(f"{class_col} = {label}: {len(class_rows)} rows")

    print("=" * 60)

    # --------------------------------------------------
    # STEP 2: Prior Probabilities
    # --------------------------------------------------
    priors = {
        label: len(class_rows) / n_total
        for label, class_rows in classes.items()
    }

    print("STEP 2: Prior Probabilities")

    for label, p in priors.items():
        print(f"P({class_col}={label}) = {len(classes[label])}/{n_total} = {p:.4f}")

    print("=" * 60)

    # --------------------------------------------------
    # STEP 3: Conditional Probabilities
    # --------------------------------------------------
    print("STEP 3: Conditional Probabilities")
    print(f"Tuple X = {new_tuple}")
    print("-" * 60)

    cond_probs = {label: [] for label in classes}

    for attr in attr_names:
        col_index = header.index(attr)
        value = new_tuple[attr]

        for label, class_rows in classes.items():
            count = sum(
                1 for row in class_rows
                if row[col_index] == value
            )

            total = len(class_rows)

            if count == 0:
                num_values = len(set(row[col_index] for row in rows))
                p = 1 / (total + num_values)
                note = " (Laplace Smoothed)"
            else:
                p = count / total
                note = ""

            cond_probs[label].append(p)

            print(
                f"P({attr}={value} | {class_col}={label}) = "
                f"{count}/{total} = {p:.4f}{note}"
            )

    print("=" * 60)

    # --------------------------------------------------
    # STEP 4: Multiply Conditional Probabilities
    # --------------------------------------------------
    print("STEP 4: Likelihood Calculation")

    likelihoods = {}

    for label, probs in cond_probs.items():
        product = 1

        for p in probs:
            product *= p

        likelihoods[label] = product

        chain = " × ".join(f"{p:.4f}" for p in probs)

        print(f"P(X | {label}) = {chain} = {product:.6f}")

    print("=" * 60)

    # --------------------------------------------------
    # STEP 5: Multiply by Prior
    # --------------------------------------------------
    print("STEP 5: Bayes Numerator")

    scores = {}

    for label in classes:
        scores[label] = likelihoods[label] * priors[label]

        print(
            f"P(X|{label}) × P({label}) = "
            f"{likelihoods[label]:.6f} × {priors[label]:.4f} "
            f"= {scores[label]:.6f}"
        )

    print("=" * 60)

    # --------------------------------------------------
    # STEP 6: Posterior Probabilities
    # --------------------------------------------------
    total_score = sum(scores.values())

    print("STEP 6: Posterior Probabilities")
    print(
        f"P(X) = {' + '.join(f'{v:.6f}' for v in scores.values())}"
        f" = {total_score:.6f}"
    )

    for label, score in scores.items():
        posterior = score / total_score

        print(
            f"P({class_col}={label} | X) = "
            f"{score:.6f}/{total_score:.6f} = {posterior:.4f}"
        )

    print("=" * 60)

    # --------------------------------------------------
    # STEP 7: Final Prediction
    # --------------------------------------------------
    prediction = max(scores, key=scores.get)

    print(f"STEP 7: FINAL PREDICTION → {class_col} = {prediction}")

    return prediction


# --------------------------------------------------
# Main Program
# --------------------------------------------------
if __name__ == "__main__":

    csv_path = input("Enter path to your CSV file: ").strip()

    header, rows = load_csv(csv_path)

    print(f"\nLoaded {len(rows)} rows from '{csv_path}'")

    new_tuple = get_new_tuple_interactively(header, rows)

    naive_bayes(header, rows, new_tuple)