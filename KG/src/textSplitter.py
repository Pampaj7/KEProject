import wikipedia
import tripletsExtractor as te

filename = "textSamples/marsDiary.csv"
filename_human = "textSamples/human.txt"


def split_text_into_chunks(text, chunk_size=1000):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def process_text(text, file, max_iterations=100):
    chunks = split_text_into_chunks(text, chunk_size=1000)
    iteration_count = 0

    for chunk in chunks:
        if iteration_count >= max_iterations:
            break
        te.modelsResponse(chunk, file)
        te.GPTResponse(chunk, file)
        iteration_count += 1


if __name__ == "__main__":
    # with open(filename, 'r') as f:
    #    full_text = f.read()
    #    process_text(full_text, "MarsDiary")

    with open(filename_human, 'r') as f:
        full_text = f.read()
        process_text(full_text, "Human")
