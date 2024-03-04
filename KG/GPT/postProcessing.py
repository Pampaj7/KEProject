from sentence_transformers import SentenceTransformer
import scipy.spatial
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Your code here

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example triplets turned into sentences for comparison
triplets_1 = ["Deep learning is a subset of machine learning methods.",
              "Deep learning is based on artificial neural networks."]
triplets_2 = ["Deep learning, subset of, machine learning methods",
              "Deep learning, based on, artificial neural networks"]

# Convert triplets to embeddings
embeddings_1 = model.encode(triplets_1)
embeddings_2 = model.encode(triplets_2)

# Calculate cosine similarity between each pair of triplets
for i, embedding_1 in enumerate(embeddings_1):
    for j, embedding_2 in enumerate(embeddings_2):
        similarity = 1 - scipy.spatial.distance.cosine(embedding_1, embedding_2)
        print(f"Similarity between triplet 1 ({i}) and triplet 2 ({j}): {similarity}")
