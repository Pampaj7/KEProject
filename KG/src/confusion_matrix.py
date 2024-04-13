import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import spacy
from tqdm import tqdm
import warnings

nlp = spacy.load("en_core_web_lg")
threshold = 0.90 # with this similarity we ha division by 0 in human text -- too small

warnings.filterwarnings("ignore", category=UserWarning, message="set_ticklabels\(\) should only be used with a fixed number of ticks")


def nlp_similarity(task):
    """
    A wrapper function to compute similarity using spaCy.
    It loads the spaCy model as needed.
    """
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")
    predicted_triplet, correct_triplets = task
    doc1 = nlp(predicted_triplet)
    for triplet_c in correct_triplets:
        doc2 = nlp(triplet_c)
        if doc1.similarity(doc2) >= threshold:
            return True
    return False


def read_triplets_from_file(file_path):
    triplets = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplet = line.strip()
            if triplet:  # Ensure the line is not empty
                triplets.add(triplet)
    return triplets


def calculate_matrix(path, similarity_score, ground_truth_file):
    correct_triplets = read_triplets_from_file(ground_truth_file)
    predicted_triplets = read_triplets_from_file("normalizedTriplets/" + path)

    print("Triplets generated", len(predicted_triplets), "Triplets GT:", len(correct_triplets))
    tasks = [(triplet_p, correct_triplets) for triplet_p in predicted_triplets]

    true_positives = 0
    with ProcessPoolExecutor() as executor: # parallel processing, may not work with very different hardware -->
        # macos fails
        futures = [executor.submit(nlp_similarity, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing predicted triplets"):
            if future.result():
                true_positives += 1

    false_positives = len(predicted_triplets) - true_positives
    false_negatives = len(correct_triplets) - true_positives

    draw_confusion_matrix(true_positives, false_positives, false_negatives, os.path.splitext(os.path.basename(path))[0])
    precision, recall, f1_score = calculate_evaluation_metrics(true_positives, false_positives, false_negatives)
    print("True Positive", true_positives, "False Positive", false_positives, "False negative", false_negatives)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    file_name = "TXT_Matrix/" + os.path.splitext(os.path.basename(path))[0] + ".txt"
    with open(file_name, 'w') as file:
        file.write("True Positives: " + str(true_positives) + "\n")
        file.write("False Positive: " + str(false_positives) + "\n")
        file.write("False negative: " + str(false_negatives) + "\n")
        file.write("Precision: " + str(precision) + "\n")
        file.write("Recall: " + str(recall) + "\n")
        file.write("F1 Score: " + str(f1_score) + "\n")
        file.write("Similarity score: " + str(similarity_score) + "\n")

    return true_positives, false_positives, false_negatives


def calculate_evaluation_metrics(tp, fp, fn):
    try:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Avoid
        # division by zero
        return precision, recall, f1_score
    except Exception as e:
        print("Error calculating evaluation metrics:", str(e))
        return 0, 0, 0  # Return zero or some other default value


def draw_confusion_matrix(true_positives, false_positives, false_negatives, name):
    """
    Draws a confusion matrix
    """
    true_negatives = 0
    confusion_matrix = np.array([[true_positives, false_positives],
                                 [false_negatives, true_negatives]])
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    # plt.show()
    plt.savefig("TXT_Matrix/" + name + ".png")