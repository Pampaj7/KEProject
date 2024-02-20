# main.py shift option f
import matplotlib.pyplot as plt
import networkx as nx
import wikipediaapi

from KGCreation import create_knowledge_graph

# he user_agent variable in the code you
# provided serves as an identifier for your application when making requests to the Wikipedia API.
# It is a common practice for APIs to require a user-agent string in HTTP requests to identify the application accessing the API.

# In this case, the user-agent string "Wikipedia-API
# Example (merlin@example.com)" is just a placeholder example.
# When you use the Wikipedia API,
# you should replace it with a user-agent string that provides some information about your application or project,
# such as its name and an optional contact email.
user_agent = "Wikipedia-API Example (merlin@example.com)"

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")

# Example usage of Wikipedia API
# This method is used to retrieve a specific page from Wikipedia.
# You provide the title of the page you're interested in as an argument. In this case, the title is "Machine learning".
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

#Ma sta roba l'ha fatta uno scemo?