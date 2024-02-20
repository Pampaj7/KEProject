import networkx as nx


def create_knowledge_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Add entities as nodes
    entities = [
        "Machine Learning",
        "Ostrava",
        "Deutsche Sprache",
        "Physics",
        "Python (in Hindu)"
    ]

    G.add_nodes_from(entities)

    # Define relationships between entities and their properties
    relationships = {
        "Machine Learning": ["Summary", "Sections", "Lang Links", "Links", "Categories"],
        "Ostrava": ["Summary", "Sections", "Lang Links", "Links", "Categories"],
        "Deutsche Sprache": ["Summary", "Sections", "Lang Links", "Links", "Categories"],
        "Physics": ["Summary", "Sections", "Lang Links", "Links", "Categories"],
        "Python (in Hindu)": ["Summary", "Sections", "Lang Links", "Links", "Categories"]
    }

    # Add relationships as edges
    for entity, properties in relationships.items():
        for prop in properties:
            G.add_edge(entity, prop)

    return G
