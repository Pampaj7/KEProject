import networkx as nx
import matplotlib.pyplot as plt

# Initialize a directed graph
G = nx.DiGraph()

# List of triplets (subject, predicate, object)
triplets = [
    ("Alan Turing", "conduct research", "field"),
    ("Alan Turing", "call", "machine intelligence"),
    ("Alan Turing", "conduct", "research"),
    ("Artificial intelligence", "found", "academic discipline"),
    ("Artificial intelligence", "go through", "cycles"),
    ("Artificial intelligence", "lead to", "spring"),
    ("Artificial intelligence", "influence", "shift"),
    ("Artificial intelligence", "raise", "questions"),
    ("AI researchers", "adapt", "techniques"),
    ("AI researchers", "integrate", "techniques"),
    ("AI researchers", "draw upon", "fields"),
    ("AI researchers", "solve", "problems"),
]

# Add nodes and edges to the graph
for subj, pred, obj in triplets:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=pred)

# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
