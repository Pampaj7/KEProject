# main.py
import logging
import matplotlib.pyplot as plt
import networkx as nx
import wikipediaapi
from KGCreation import create_knowledge_graph

logging.basicConfig(level=logging.INFO)

user_agent = "Wikipedia-API Example (merlin@example.com)"

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")

# Example usage of Wikipedia API
page_py = wiki_wiki.page("Machine learning")

print("Page - Exists: %s" % page_py.exists())
print("Page - Id: %s" % page_py.pageid)
print("Page - Title: %s" % page_py.title)
print("Page - Summary: %s" % page_py.summary[0:60])

# Creating knowledge graph
G = create_knowledge_graph()

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # Set a seed for reproducibility
nx.draw(G, pos, with_labels=True, node_size=2000,
        node_color="lightblue", font_size=10, font_weight="bold")
plt.title("Simplified Knowledge Graph")
plt.show()
