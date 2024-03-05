import wikipedia
import tripletsExtractor as te

filename = "testi/porn.txt"


def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def process_text(text):
    chunks = split_text_into_chunks(text, chunk_size=100)

    for chunk in chunks:
        # Here, you call GPTResponse or modelsResponse for each chunk
        # Assuming GPTResponse is adapted to handle a text chunk and store its output
        te.modelsResponse(chunk)
        # modelsResponse could be similarly adapted if needed


if __name__ == "__main__":
    with open(filename, 'r') as f:
        full_text = f.read()

    process_text(full_text)  # Process the text in chunks
