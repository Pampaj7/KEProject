import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from scipy.linalg import eigh
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

np.random.seed(42)

def calculate_and_plot_metrics(G, title):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    print(f"[{title}] Number of nodes: {nodes}")
    print(f"[{title}] Number of edges: {edges}")
    print(f"[{title}] Density: {density:.4f}")

    # Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    avg_degree = np.mean(degrees)
    print(f"[{title}] Average degree: {avg_degree:.2f}")

    plt.figure(figsize=(8, 4))
    plt.hist(degrees, bins=20, alpha=0.5, label=title)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title(f'Degree Distribution - {title}')
    plt.legend()
    plt.show()


# make the graph as a vector
def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=8, seed=42)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return embeddings


# compute the cosine similarity between the embeddings
def compare_embeddings(embeddings1, embeddings2):
    similarity = cosine_similarity(embeddings1, embeddings2)
    avg_similarity = np.mean(similarity)
    print(f"Average Cosine Similarity: {avg_similarity}")
    return avg_similarity


def visualize_embeddings(embeddings, title):
    """
    Visualize embeddings using t-SNE for dimensionality reduction.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title(f't-SNE visualization of {title}')
    plt.show()


def test_graphs(g0, g1, filename):
    calculate_and_plot_metrics(g1, filename)
    embeddings = generate_embeddings(g1)
    cosine_similarity = compare_embeddings(g0, embeddings)
    visualize_embeddings(embeddings, filename)
    return cosine_similarity