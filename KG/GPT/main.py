import KG as kg
import graph_stats as gs
import confusion_matrix as cm


def process_files(files):
    graphs = []
    cosine_sim = []

    gpt_graph = kg.KG_creation(files[0])
    gpt_embeddings = gs.generate_embeddings(gpt_graph)

    for file in files[1:]:
        print("RUNNING FILE: ", file)
        g = kg.KG_creation(file)
        graphs.append(g)
        current_embeddings = gs.generate_embeddings(g)
        similarity_score = gs.compare_embeddings(gpt_embeddings, current_embeddings)
        cosine_sim.append(similarity_score)
        cm.calculate_matrix(file, similarity_score)


filename = [
    "extracted_text_from_GPT.txt",
    "extracted_text_from_Mistral7B_CME_v1_LOCAL.txt",
    "extracted_text_from_llama-13b-chat.txt",
    "extracted_text_from_llama-70b-chat.txt",
    "extracted_text_from_mistral-7b-instruct.txt",
    "extracted_text_from_mixtral-8x7b-instruct.txt",
    "extracted_text_from_vicuna-13b.txt"
]

filename_human = [
    "extracted_text_from_human.txt",
    "extracted_text_from_llama-13b-chat_Human.txt",
    "extracted_text_from_llama-70b-chat_Human.txt",
    "extracted_text_from_mistral-7b-instruct_Human.txt",
    "extracted_text_from_mixtral-8x7b-instruct_Human.txt",
    "extracted_text_from_vicuna-13b_Human.txt"
]

process_files(filename)
process_files(filename_human)

# some basic interrogation
# oi.interrogate()
