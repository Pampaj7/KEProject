from rdflib import Graph


def interrogate():
    g = Graph()
    try:
        g.parse("turtle/ontology__normalizedextracted_text_from_GPT_MarsDiary.ttl", format="ttl")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Example SPARQL Queries

    # Query 1: Find objects of a specific predicate
    predicate_to_find = 'http://example.org/myontology/mapped'  # replace with actual predicate
    query3 = f"""
    SELECT DISTINCT ?o
    WHERE {{
      ?s <{predicate_to_find}> ?o .
    }}
    """
    print(f"\nObjects for predicate {predicate_to_find}:")
    for row in g.query(query3):
        print(row.o)
