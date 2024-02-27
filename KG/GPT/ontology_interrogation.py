from rdflib import Graph


def interrogate():
    g = Graph()
    g.parse("turtle/ontology_extracted_text_from_GPT.ttl", format="ttl")

    # Assuming g is your RDFLib Graph containing the ontology
    sparql_query = """
    PREFIX ns1: <http://example.org/myontology/>
    
    SELECT ?subject ?application
    WHERE {
      ?subject ns1:issoldas ?app .
      BIND(STRAFTER(STR(?app), STR(ns1:)) AS ?application)
    }
    """

    # Execute the query
    for row in g.query(sparql_query):
        print(f"Application: {row.application}")
