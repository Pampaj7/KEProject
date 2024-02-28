import tripletsExtractor as te
import KG as kg
import ontology_interrogation as oi
import graph_stats as gs

filename = ["extracted_text_from_GPT.txt", "extracted_text_from_llama-13b-chat.txt",
            "extracted_text_from_llama-70b-chat.txt",
            "extracted_text_from_mistral-7b-instruct.txt", "extracted_text_from_mixtral-8x7b-instruct.txt",
            "extracted_text_from_vicuna-13b.txt"]

# Extract triplets from Wikipedia
topic = "deep learning"
text_from_wikipedia = te.get_wikipedia_text(topic)
te.modelsResponse(text_from_wikipedia)
te.GPTResponse(topic)

# Create KG from triplets
for file in filename:
    g = kg.KG_creation(file)
    gs.calculate_and_plot_metrics(g, file)
    #gs.plot_spectrum(g, file)

oi.interrogate()
