import matplotlib.pyplot as plt
import numpy as np


def read_triplets_from_file(file_path):
    """
    Reads triplets from a given file path.
    Each triplet is expected to be on a separate line in the format: subject, relation, object.
    """
    triplets = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplet = line.strip()
            if triplet:  # Ensure the line is not empty
                triplets.add(triplet)
    return triplets


def calculate_matrix(path):
    correct_triplets = read_triplets_from_file("normalizedTriplets/_normalizedextracted_text_from_GPT.txt")
    predicted_triplets = read_triplets_from_file("normalizedTriplets/_normalized" + path)

    true_positives = len(correct_triplets.intersection(predicted_triplets))
    false_positives = len(predicted_triplets.difference(correct_triplets))
    false_negatives = len(correct_triplets.difference(predicted_triplets))

    draw_confusion_matrix(true_positives, false_positives, false_negatives)
    """precision, recall, f1_score = calculate_evaluation_metrics(true_positives, false_positives, false_negatives)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")"""
    #TODO can work but need to postprocess the text

    return true_positives, false_positives, false_negatives


def calculate_evaluation_metrics(true_positives, false_positives, false_negatives):
    """
    Calculates precision, recall, and F1 score from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def draw_confusion_matrix(true_positives, false_positives, false_negatives):
    """
    Draws a confusion matrix using Matplotlib.
    """
    # Calculate true negatives (assuming it's not directly relevant and set to 0 for visualization)
    true_negatives = 0

    # Create a confusion matrix
    confusion_matrix = np.array([[true_positives, false_positives],
                                 [false_negatives, true_negatives]])

    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticklabels([''] + ['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels([''] + ['Actual Positive', 'Actual Negative'])

    # Display the numbers in the cells
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
