import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import spacy
from tqdm import tqdm
import warnings

nlp = spacy.load("en_core_web_lg")
threshold = 0.95

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


def calculate_matrix(path, similarity_score):
    correct_triplets = read_triplets_from_file("normalizedTriplets/_normalizedextracted_text_from_GPT.txt")
    predicted_triplets = read_triplets_from_file("normalizedTriplets/_normalized" + path)

    print("Triplets generated", len(predicted_triplets), "Triplets GT:", len(correct_triplets))
    tasks = [(triplet_p, correct_triplets) for triplet_p in predicted_triplets]

    true_positives = 0
    with ProcessPoolExecutor() as executor:
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


def calculate_evaluation_metrics(true_positives, false_positives, false_negatives):
    """
    Calculates precision, recall, and F1 score from true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def draw_confusion_matrix(true_positives, false_positives, false_negatives, name):
    """
    Draws a confusion matrix using Matplotlib.
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