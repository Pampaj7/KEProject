import KG as kg
import graph_stats as gs
import confusion_matrix as cm

filename = ["extracted_text_from_GPT.txt", "extracted_text_from_llama-13b-chat.txt",
            "extracted_text_from_llama-70b-chat.txt",
            "extracted_text_from_mistral-7b-instruct.txt", "extracted_text_from_mixtral-8x7b-instruct.txt",
            "extracted_text_from_vicuna-13b.txt"]

# Extract triplets from Wikipedia
# Split extraction to kg creation
# topic = "deep learning"
# text_from_wikipedia = te.get_wikipedia_text(topic)
# te.modelsResponse(text_from_wikipedia)
# te.GPTResponse(text_from_wikipedia)

# Create KG from triplets
graphs = []
cosine_sim = []

gpt_graph = kg.KG_creation(filename[0])
gpt_embeddings = gs.generate_embeddings(gpt_graph)

for file in filename[1:]:
    g = kg.KG_creation(file)
    graphs.append(g)
    current_embeddings = gs.generate_embeddings(g)
    similarity_score = gs.compare_embeddings(gpt_embeddings, current_embeddings)
    cosine_sim.append(similarity_score)
    cm.calculate_matrix(file)

similarity_percentages = [score * 100 for score in cosine_sim]

for file, score, percentage in zip(filename, cosine_sim, similarity_percentages):
    print(f"File: {file}, Cosine Similarity: {score:.4f}, Similarity Percentage: {percentage:.2f}%")

# some basic interrogation
# oi.interrogate()
