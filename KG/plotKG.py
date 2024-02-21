import pydot
import rdflib
from rdflib import Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
import matplotlib.pyplot as plt

g = Graph()
result = g.parse("firstpack.json", format="json-ld")
#result = g.parse("firstpack.json", format="json-ld")

for subj, pred, obj in g:
    if (subj, pred, obj) not in g:
       raise Exception("It better be!")

print(f"Graph g has {len(g)} statements.")
print(g.serialize(format="turtle"))

G = rdflib_to_networkx_multidigraph(result)

# Plot Networkx instance of RDF Graph
pos = nx.spring_layout(G, k = 0.5)
edge_labels = nx.get_edge_attributes(G, 'c' )
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.draw(G, with_labels=False, node_size=30, font_weight='bold', edge_cmap=plt.cm.Blues, pos = pos)

#if not in interactive mode for
plt.show()


