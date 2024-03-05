import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from scipy.linalg import eigh
from node2vec import Node2Vec
import matplotlib.pyplot as plt


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



def check_isomorphism(G1, G2):

    matcher = isomorphism.GraphMatcher(G1, G2)
    are_isomorphic = matcher.is_isomorphic()
    print(f"Graph 1 and Graph 2 are isomorphic: {are_isomorphic}")


def plot_spectrum(L, title):

    eigenvalues = np.sort(eigh(L, eigvals_only=True))
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues, marker='o')
    plt.title(f'Eigenvalues of Laplacian Matrix - {title}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()


def generate_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return embeddings


def compare_embeddings(embeddings1, embeddings2):
    similarity = cosine_similarity(embeddings1, embeddings2)
    avg_similarity = np.mean(similarity)
    print(f"Average Cosine Similarity: {avg_similarity}")


def analyze_connected_components(G, title):
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    print(f"[{title}] Number of Connected Components: {len(connected_components)}")
    print(f"[{title}] Largest Connected Component Size: {len(largest_cc)}")
    # if print the same thing is because the try pre-existing graphs


def detect_and_analyze_communities(G, title):
    partition = community_louvain.best_partition(G)  # TODO should work check lib conflicts
    num_communities = len(set(partition.values()))
    print(f"[{title}] Number of Communities Detected: {num_communities}")

    # Visualize the community structure
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title(f"Community Structure - {title}")
    plt.show()


def visualize_embeddings(embeddings, title):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title(f"Node Embeddings Visualization - {title}")
    plt.show()



def test():
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

    analyze_connected_components(G1, "Graph 1")
    analyze_connected_components(G2, "Graph 2")

    # detect_and_analyze_communities(G1, "Graph 1")
    # detect_and_analyze_communities(G2, "Graph 2")

    visualize_embeddings(embeddings_G1, "Graph 1")  # trash
    visualize_embeddings(embeddings_G2, "Graph 2")


