import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"For k = {model.n_neighbors}:")
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = list()
    labels = [] # both mean list
    month_lookup = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }
    visitor_type_lookup = {
        "Returning_Visitor": 1,
        "New_Visitor": 0,
        "Other": 0
    }
    bool_lookup = {
        "TRUE": 1,
        "FALSE": 0
    }
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence_row = []

            evidence_row.append(int(row["Administrative"]))
            evidence_row.append(float(row["Administrative_Duration"]))
            evidence_row.append(int(row["Informational"]))
            evidence_row.append(float(row["Informational_Duration"]))
            evidence_row.append(int(row["ProductRelated"]))
            evidence_row.append(float(row["ProductRelated_Duration"]))
            evidence_row.append(float(row["BounceRates"]))
            evidence_row.append(float(row["ExitRates"]))
            evidence_row.append(float(row["PageValues"]))
            evidence_row.append(float(row["SpecialDay"]))

            month = month_lookup[row["Month"]]
            evidence_row.append(month)

            evidence_row.append(int(row["OperatingSystems"]))
            evidence_row.append(int(row["Browser"]))
            evidence_row.append(int(row["Region"]))
            evidence_row.append(int(row["TrafficType"]))

            visitor = visitor_type_lookup[row["VisitorType"]]
            evidence_row.append(visitor)

            weekend = bool_lookup[row["Weekend"]]
            evidence_row.append(weekend)

            label = bool_lookup[row["Revenue"]]
            labels.append(label)
            evidence.append(evidence_row)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model trained on the data.
    """
    k = 3
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(evidence, labels)
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    count = min(len(labels), len(predictions))

    actual_positive_count = sum(
        1 for i in range(count)
        if labels[i] == 1)

    actual_negative_count = count - actual_positive_count

    true_positives = sum(
        1 for i in range(count)
        if labels[i] == 1 and predictions[i] == 1
    )

    true_negatives = sum(
        1 for i in range(count)
        if labels[i] == 0 and predictions[i] == 0
    )


    # correctly predicted positives / actual number of positives
    sensitivity = None
    if actual_positive_count:
        sensitivity = true_positives / actual_positive_count

    # correctly predicted negatives / actual number of negatives
    specificity = None
    if actual_negative_count:
        specificity = true_negatives / actual_negative_count

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
