from rdflib import Graph


def interrogate():
    g = Graph()
    try:
        g.parse("turtle/ontology__normalizedextracted_text_from_llama-13b-chat_MarsDiary.ttl", format="ttl")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

        # Query to find what Venus has according to the ontology
    sparql_query_venus_attributes = """
       PREFIX ns2: <http://example.org/myontology/>

       SELECT ?attribute ?value
       WHERE {
         ns2:_Venus ns2:has ?attribute .
         OPTIONAL { ?attribute ns2: ?value }
       }
       """
    for row in g.query(sparql_query_venus_attributes):
        attribute = row.attribute.split("#")[-1] if "#" in row.attribute else row.attribute
        print(f"Venus has: {attribute}, Value: {row.value if row.value else 'No value provided'}")
