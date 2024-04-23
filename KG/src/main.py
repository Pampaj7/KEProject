import KG as kg
import graph_stats as gs
import confusion_matrix as cm
import ontology_interrogation as oi

GTai = "normalizedTriplets/_normalizedextracted_text_from_GPT_MarsDiary.txt"
GThuman = "normalizedTriplets/extracted_text_from_human.txt"  #handmade


def process_files(files, gtfiles):
    graphs = []
    cosine_sim = []

    gpt_graph = kg.KG_creation(files[0])  # first text is the GT
    gpt_embeddings = gs.generate_embeddings(gpt_graph)

    for file in files[1:]:
        print("RUNNING FILE: ", file)
        g = kg.KG_creation(file)
        graphs.append(g)
        current_embeddings = gs.generate_embeddings(g)
        if current_embeddings is None:
            continue
        similarity_score = gs.compare_embeddings(gpt_embeddings, current_embeddings)
        cosine_sim.append(similarity_score)
        cm.calculate_matrix(file, similarity_score, gtfiles)  #output matrix TT ecc is calculated with cos sim


filename = [
    "_normalizedextracted_text_from_GPT_MarsDiary.txt",
    "_normalizedextracted_text_from_Mistral7B_CME_v1_LOCAL_marsDiary.txt",
    "_normalizedextracted_text_from_llama-13b-chat_MarsDiary.txt",
    "_normalizedextracted_text_from_llama-70b-chat_MarsDiary.txt",
    "_normalizedextracted_text_from_mistral-7b-instruct_MarsDiary.txt",
    "_normalizedextracted_text_from_mixtral-8x7b-instruct_MarsDiary.txt",
    "_normalizedextracted_text_from_vicuna-13b_MarsDiary.txt"
]

filename_human = [
    "extracted_text_from_human.txt", #handmande
    "_normalizedextracted_text_from_GPT_Human.txt" # was missing
    "_normalizedextracted_text_from_Mistral7B_CME_v1_LOCAL_human.txt",
    "_normalizedextracted_text_from_llama-13b-chat_Human.txt",
    "_normalizedextracted_text_from_llama-70b-chat_Human.txt",
    "_normalizedextracted_text_from_mistral-7b-instruct_Human.txt",
    "_normalizedextracted_text_from_mixtral-8x7b-instruct_Human.txt",
    "_normalizedextracted_text_from_vicuna-13b_Human.txt"
]

process_files(filename, GTai)
process_files(filename_human, GThuman)

oi.interrogate()  # just to show it works
