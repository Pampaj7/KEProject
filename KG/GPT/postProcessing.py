import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


word = "car"
synonyms = get_synonyms(word)
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_closest_embeddings(query_embedding, embeddings, labels, top_n=5):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:]
    return [(labels[i], similarities[i]) for i in top_indices[::-1]]


syn_list = []
for syn in synonyms:
    syn_list.append(syn)

labels = syn_list
embeddings = np.array([model.encode(label) for label in labels])
query_embedding = model.encode(word)  # basically is the cosine similarity between the synonyms of car and the word car
# can be used in the calculate_matrix function in confusion_matrix.py to calculate the similarity between the predicted
# and the correct triplets

closest_synonyms = find_closest_embeddings(query_embedding, embeddings, labels)
print(closest_synonyms)
