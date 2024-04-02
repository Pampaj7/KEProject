import wikipedia
import tripletsExtractor as te

filename = "testi/Mistral7B_CME_v1.csv"


def split_text_into_chunks(text, chunk_size=1000):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def process_text(text):
    chunks = split_text_into_chunks(text, chunk_size=1000)

    for chunk in chunks:
        te.modelsResponse(chunk)
        te.GPTResponse(chunk)
        # modelsResponse could be similarly adapted if needed


if __name__ == "__main__":
    with open(filename, 'r') as f:
        full_text = f.read()

    process_text(full_text)  # Process the text in chunks
