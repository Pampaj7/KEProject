import tripletsExtractor as te
import KG as kg
import ontology_interrogation as oi
import graph_stats as gs
import confusion_matrix as cm

filename = ["extracted_text_from_GPT.txt", "extracted_text_from_llama-13b-chat.txt",
            "extracted_text_from_llama-70b-chat.txt",
            "extracted_text_from_mistral-7b-instruct.txt", "extracted_text_from_mixtral-8x7b-instruct.txt",
            "extracted_text_from_vicuna-13b.txt"]

# Extract triplets from Wikipedia
# Split extraction to kg creation
#topic = "deep learning"
#text_from_wikipedia = te.get_wikipedia_text(topic)
#te.modelsResponse(text_from_wikipedia)
#te.GPTResponse(text_from_wikipedia)

# Create KG from triplets
graphs = []
for file in filename:
    g = kg.KG_creation(file)
    graphs.append(g)
    gs.test_graphs(gs.generate_embeddings(graphs[0]), g, file)
    cm.calculate_matrix(file)

#some basic interrogation
#oi.interrogate()
