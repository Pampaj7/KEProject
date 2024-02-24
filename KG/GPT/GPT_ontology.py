from rdflib import Graph, URIRef, RDF, RDFS, Namespace

# Define your namespace
my_namespace = Namespace("http://example.org/myontology/")

# Initialize a graph
g = Graph()

# Define and add classes to graph
deep_learning_class = URIRef(my_namespace.DeepLearning)
machine_learning_methods_class = URIRef(my_namespace.MachineLearningMethods)
g.add((deep_learning_class, RDF.type, RDFS.Class))
g.add((machine_learning_methods_class, RDF.type, RDFS.Class))

# Define and add property to graph
is_a_subset_of = URIRef(my_namespace.isASubsetOf)
g.add((is_a_subset_of, RDF.type, RDF.Property))
g.add((is_a_subset_of, RDFS.domain, deep_learning_class))
g.add((is_a_subset_of, RDFS.range, machine_learning_methods_class))

# Add individuals and their relationships
deep_learning_individual = URIRef(my_namespace.DeepLearningInstance)
machine_learning_methods_individual = URIRef(my_namespace.MachineLearningMethodsInstance)
g.add((deep_learning_individual, RDF.type, deep_learning_class))
g.add((machine_learning_methods_individual, RDF.type, machine_learning_methods_class))
g.add((deep_learning_individual, is_a_subset_of, machine_learning_methods_individual))

# Adding a new instance as an example
new_instance = URIRef(my_namespace.NewDeepLearningInstance)
g.add((new_instance, RDF.type, deep_learning_class))
g.add((new_instance, is_a_subset_of, machine_learning_methods_individual))

# SPARQL query to retrieve all classes
query_classes = """
SELECT ?class
WHERE {
  ?class a rdfs:Class .
}
"""

# Execute the query for classes
print("Classes in the Ontology:")
for row in g.query(query_classes):
    print(f"Class found: {row[0]}")

# SPARQL query to retrieve the relationships of the DeepLearningInstance
query_relationships = f"""
PREFIX ns1: <{my_namespace}>
SELECT ?property ?value
WHERE {{
  ns1:DeepLearningInstance ?property ?value .
}}
"""

# Execute the query for relationships
print("\nRelationships of DeepLearningInstance:")
for row in g.query(query_relationships):
    print(f"Property: {row.property}, Value: {row.value}")

# Serialize and print the graph to Turtle format
print("\nUpdated RDF Graph in Turtle format:")
print(g.serialize(format='turtle').encode("utf-8"))
