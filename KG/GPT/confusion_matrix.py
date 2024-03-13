import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_lg")
threshold = 0.96


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
    title = path.split("_")[-1].split(".")[0]

    print("Triplets generated:", len(predicted_triplets), "Triplets GT:", len(correct_triplets))

    tasks = [(triplet_p, correct_triplets) for triplet_p in predicted_triplets]

    true_positives = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(nlp_similarity, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing predicted triplets"):
            if future.result():
                true_positives += 1

    false_positives = len(predicted_triplets) - true_positives
    false_negatives = len(correct_triplets) - true_positives

    draw_confusion_matrix(true_positives, false_positives, false_negatives, title)
    precision, recall, f1_score = calculate_evaluation_metrics(true_positives, false_positives, false_negatives)

    print("True Positive:", true_positives, "False Positive:", false_positives, "False Negative:", false_negatives)
    print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1_score:.2f}")

    return true_positives, false_positives, false_negatives


def calculate_evaluation_metrics(true_positives, false_positives, false_negatives):
    """
    Calculates precision, recall, and F1 score from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def draw_confusion_matrix(true_positives, false_positives, false_negatives, title):
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
    plt.title('Confusion Matrix for ' + title)
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
