import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
import numpy as np
from scipy.linalg import eigh
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity


def calculate_and_plot_metrics(G, title):
    """
    Calculate and print graph metrics, and plot the degree distribution.
    """
    # Basic metrics
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

    # Clustering coefficient
    clustering = nx.average_clustering(G)
    print(f"[{title}] Average Clustering Coefficient: {clustering:.4f}")

    # Diameter and Average Path Length (for connected components)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"[{title}] Diameter: {diameter}")
        print(f"[{title}] Average Path Length: {avg_path_length:.4f}")
    else:
        print(f"[{title}] Graph is not fully connected; consider component-wise metrics.")


def check_isomorphism(G1, G2):
    """
    Check if two graphs are isomorphic and print the result.
    """
    matcher = isomorphism.GraphMatcher(G1, G2)
    are_isomorphic = matcher.is_isomorphic()
    print(f"Graph 1 and Graph 2 are isomorphic: {are_isomorphic}")


def plot_spectrum(L, title):
    """
    Plot the spectrum of a Laplacian matrix.
    """
    eigenvalues = np.sort(eigh(L, eigvals_only=True))
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues, marker='o')
    plt.title(f'Eigenvalues of Laplacian Matrix - {title}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()


# Generate node embeddings for two graphs
def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return embeddings


# Compare two sets of embeddings
def compare_embeddings(embeddings1, embeddings2):
    # Simple comparison: average cosine similarity between corresponding nodes
    # Note: Assumes node sets of G1 and G2 are the same. For different node sets, additional steps are needed.
    similarity = cosine_similarity(embeddings1, embeddings2)
    avg_similarity = np.mean(similarity)
    print(f"Average Cosine Similarity: {avg_similarity}")


# Example usage:
G1 = nx.gnp_random_graph(100, 0.1, seed=42)
G2 = nx.gnp_random_graph(100, 0.5, seed=42)

calculate_and_plot_metrics(G1, "Graph 1")
calculate_and_plot_metrics(G2, "Graph 2")

check_isomorphism(G1, G2)

# Spectral analysis
L1 = nx.laplacian_matrix(G1).toarray()
L2 = nx.laplacian_matrix(G2).toarray()
plot_spectrum(L1, "Graph 1")
plot_spectrum(L2, "Graph 2")

embeddings_G1 = generate_embeddings(G1)
embeddings_G2 = generate_embeddings(G2)

compare_embeddings(embeddings_G1, embeddings_G2)


