import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, RDF, Namespace
import re

filename = ['GPT-4model.txt', "extracted_text_from_llama.txt"]

def normalize_name(name):
    # Replace spaces with underscores and remove < and > characters
    name = name.replace(" ", "_").replace("<", "").replace(">", "").replace(".", "")
    # Add more normalization as needed, e.g., removing or replacing other special characters
    name = re.sub(r'\d+', '', name)

    return name

def create_ontology(triplets, filename):
    # Initialize RDF graph
    g = Graph()

    # Define a namespace for the ontology
    ns = Namespace("http://example.org/myontology/")

    # Add triplets to the RDF graph
    for subject, predicate, object in triplets:
        normalized_subject = normalize_name(subject)
        normalized_predicate = normalize_name(predicate)  # Assuming predicate is the object in your context
        normalized_object = normalize_name(object)

        s = URIRef(ns[normalized_subject])
        p = URIRef(ns[normalized_predicate])
        o = URIRef(ns[normalized_object])
        g.add((s, p, o))

    # Serialize the graph to a file (Turtle format is a common choice for ontologies)
    filename_without_extension = filename.replace(".txt", "")


    g.serialize(destination="ontology_" + filename_without_extension + ".ttl", format="turtle")

    print("Ontology created and saved to 'ontology_" + filename_without_extension + ".ttl")


def KG_creation(filename):
    triplets = []
    with open(filename, 'r') as file:
        for line in file:
            cleaned_line = line.strip().strip('<>')
            parts = cleaned_line.split(', ')
            if len(parts) == 3:
                triplets.append(parts)

    #draw the graph
    G = nx.DiGraph()
    for subject, relation, predicate in triplets:
        G.add_edge(subject, predicate, relation=relation)

    # Choose a layout
    pos = nx.spring_layout(G, k=0.5)  # k adjusts the distance between nodes

    # Draw the graph
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500,
            edge_color='gray', linewidths=0.5, font_size=10)

    # Draw edge labels
    edge_labels = dict([((u, v,), d['relation'])
                        for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)

    plt.title('Knowledge Graph Visualization')
    plt.axis('off')
    filename_without_extension = filename.replace(".txt", "")
    plt.savefig('knowledge_graph' + filename_without_extension + '.png')
    plt.show()

    # Now also create the ontology
    create_ontology(triplets, filename)



KG_creation(filename[0])
KG_creation(filename[1])
