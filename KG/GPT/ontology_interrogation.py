from rdflib import Graph


def interrogate():
    g = Graph()
    g.parse("turtle/ontology_extracted_text_from_GPT.ttl", format="ttl")

    sparql_query_deep_based_on = """
    PREFIX ns1: <http://example.org/myontology/>

    SELECT ?basedOn
    WHERE {
      ns1:Deeplearning ns1:basedon ?basedOn .
    }
    """

    for row in g.query(sparql_query_deep_based_on):
        print(f"Deep Learning is based on: {row.basedOn}")

    sparql_query_applications = """
    PREFIX ns1: <http://example.org/myontology/>

    SELECT ?application
    WHERE {
      ns1:Deeplearningarchitectures ns1:havebeenappliedto ?application .
    }
    """

    for row in g.query(sparql_query_applications):
        print(f"Application: {row.application}")

    sparql_query_inspirations = """
    PREFIX ns1: <http://example.org/myontology/>

    SELECT ?inspiration
    WHERE {
      ns1:Artificialneuralnetworks ns1:wereinspiredby ?inspiration .
    }
    """

    for row in g.query(sparql_query_inspirations):
        print(f"Inspiration: {row.inspiration}")

    sparql_query_characteristics = """
    PREFIX ns1: <http://example.org/myontology/>

    SELECT ?characteristic
    WHERE {
      ns1:Artificialneuralnetworks ns1:tendtobe ?characteristic .
    }
    """

    for row in g.query(sparql_query_characteristics):
        print(f"Characteristic: {row.characteristic}")


