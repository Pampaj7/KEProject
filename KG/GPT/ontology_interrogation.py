from rdflib import Graph

g = Graph()
g.parse("ontology_GPT-4model.ttl", format="ttl")

# Assuming g is your RDFLib Graph containing the ontology
sparql_query = """
PREFIX ns1: <http://example.org/myontology/>

SELECT ?subject ?application
WHERE {
  ?subject ns1:basedon ?app .
  BIND(STRAFTER(STR(?app), STR(ns1:)) AS ?application)
}
"""

# Execute the query
for row in g.query(sparql_query):
    print(f"Application: {row.application}")
