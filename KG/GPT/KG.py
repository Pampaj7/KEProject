import networkx as nx
import matplotlib.pyplot as plt

filename = ['GPT-4model.txt', "extracted_text_from_llama.txt"]


def KG_creation(filename):
    triplets = []
    with open(filename, 'r') as file:
        for line in file:
            cleaned_line = line.strip().strip('<>')
            parts = cleaned_line.split(', ')
            if len(parts) == 3:
                triplets.append(parts)
    G = nx.DiGraph()

    for subject, relation, predicate in triplets:
        G.add_edge(subject, predicate, relation=relation)

    # Step 3: Visualize the Knowledge Graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G)  # k controls the spacing between nodes
    nx.draw(G, pos, with_labels=True, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='skyblue', alpha=0.9,
            labels={node: node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['relation'] for u, v, d in G.edges(data=True)},
                                 label_pos=0.3, font_size=9)
    plt.title('Knowledge Graph Visualization')
    plt.axis('off')
    plt.show()


KG_creation(filename[0])
KG_creation(filename[1])
