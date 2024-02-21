from rdflib import Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


def load_rdf_graph(file_path, format="json-ld"):
    """Load RDF graph from a file."""
    g = Graph()
    g.parse(file_path, format=format)
    print(f"Graph has {len(g)} statements.")
    return g


def visualize_with_matplotlib(G):
    """Visualize the RDF graph using matplotlib."""
    node_degrees = dict(G.degree())
    most_important_node = max(node_degrees, key=node_degrees.get)

    pos = nx.spring_layout(G, k=0.5)  # Compute layout
    degrees = dict(nx.degree(G))
    # Draw nodes and edges
    # Draw all nodes and edges
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='skyblue', edge_color='k')

    # Draw the most important node with a distinct style and label
    nx.draw_networkx_nodes(G, pos, nodelist=[most_important_node], node_size=100, node_color='red')
    nx.draw_networkx_labels(G, pos, labels={most_important_node: f'Node {most_important_node}'}, font_color='red')

    plt.show()


def visualize_with_pyvis(g, output_file="rdf_graph.html"):
    """Convert RDF graph to a pyvis network graph and visualize it."""
    nt = Network("500px", "1000px", notebook=False)  # Adjust size as needed
    # Add nodes and edges
    for subj, pred, obj in g:
        nt.add_node(str(subj), title=str(subj))
        nt.add_node(str(obj), title=str(obj))
        nt.add_edge(str(subj), str(obj), title=str(pred))
    # Set visualization options
    nt.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0
        },
        "minVelocity": 0.75
      }
    }
    """)
    nt.show(output_file, notebook=False)


# Load RDF graph
rdf_graph = load_rdf_graph("knowledge_graph.json")

# Convert to NetworkX graph for matplotlib visualization
G = rdflib_to_networkx_multidigraph(rdf_graph)

# Visualize using matplotlib
visualize_with_matplotlib(G)

# Visualize using pyvis for an interactive graph
visualize_with_pyvis(rdf_graph)
