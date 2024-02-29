import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, RDF, Namespace
import re
import os
from pyvis.network import Network


def normalize_name(name):
    # Replace spaces with unpleasant characters
    name = (name.replace(" ", "").replace("<", "").replace(">", "").
            replace(".", "").replace("-", "").replace("'", "").
            replace('"', "").replace("(", "").replace(")", "").replace(",", "").replace(":", "").replace(";", ""))
    name = re.sub(r'\d+', '', name)

    return name


def create_ontology(triplets, filename):
    # !!! this is a graph from rdf lib, not networkx
    g = Graph()

    # Define a namespace for the ontology
    ns = Namespace("http://example.org/myontology/")

    # Add triplets to the RDF graph
    for subject, predicate, object in triplets:
        normalized_subject = normalize_name(subject)
        normalized_predicate = normalize_name(predicate)
        normalized_object = normalize_name(object)

        s = URIRef(ns[normalized_subject])
        p = URIRef(ns[normalized_predicate])
        o = URIRef(ns[normalized_object])
        g.add((s, p, o))

    # Serialize the graph to a turtle file
    filename_without_extension = filename.replace(".txt", "")

    g.serialize(destination="turtle/ontology_" + filename_without_extension + ".ttl", format="turtle")

    print("Ontology created and saved to 'ontology_" + filename_without_extension + ".ttl")


def KG_creation(filename):
    triplets = []
    filename_without_extension = filename.replace(".txt", "").replace(".txt", "")

    full_path = os.path.join("tripletsTXTs/", filename)

    with open(full_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip().strip('<>')
            parts = cleaned_line.split(', ')
            if len(parts) == 3:
                # Normalize names for each part of the triplet -- works all the time
                normalized_subject = normalize_name(parts[0])
                normalized_relation = normalize_name(parts[1])
                normalized_object = normalize_name(parts[2])
                triplets.append((normalized_subject, normalized_relation, normalized_object))

    G = nx.DiGraph()  # networkx graph
    for subject, relation, object in triplets:
        # Names are already normalized in the triplets
        G.add_edge(subject, object, relation=relation)

    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000,
            edge_color='gray', linewidths=0.7, font_size=14)

    # Draw edge labels
    edge_labels = dict([((u, v,), d['relation'])
                        for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)
    plt.title('Knowledge Graph Visualization: ' + filename_without_extension)
    plt.axis('off')
    plt.savefig('plots/knowledge_graph_' + filename_without_extension + '.png')

    # Now also create the ontology with normalized names
    create_ontology(triplets, filename)

    #interactive visualization
    visualize_with_pyvis(G, 'knowledge_graph' + filename_without_extension + '.html', filename_without_extension)

    return G


def visualize_with_pyvis(g, output_file, filename_without_extension):
    """Convert networkx graph to a pyvis network graph and visualize it."""
    nt = Network("1200px", "1600px", notebook=True)  # if problem with this line try change notebook = false
    # networkx graph to pyvis network conversion
    nt.from_nx(g)
    nt.save_graph('plots/knowledge_graph_' + filename_without_extension + '.html')
