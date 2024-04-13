import wikipedia
from openai import OpenAI
import os
import openai
from wikipedia import wikipedia
from langchain_community.chat_models import ChatOpenAI
import KG as kg

# Configuration settings
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


def GPTResponse(testimony, file):
    prompt_template = """
    You will perform the open information extraction task. You will identify the named entities in the content and then extract the relations between them.
    Based on the provided testimony, you will return triples which are formatted as <named entity A, relation, named entity B> without enumerating them.

    START of the testimony:
    {testimony}
    END of the testimony.

    The extracted triples formatted as <named entity A, relation, named entity B> are:
    """
    prompt = prompt_template.format(testimony=testimony)

    chat = ChatOpenAI(model_name="gpt-4", temperature=0)  # drains cash
    res = chat.predict(prompt)
    filename = f'extracted_text_from_GPT_{file}.txt'
    save_text_to_file(res, filename, "tripletsTXTs")
    kg.normalize_file("tripletsTXTs/" + filename,
                      "normalizedTriplets/_normalized" + filename)


def modelsResponse(text, file):
    output_directory_not_norm = "tripletsTXTs"
    output_directory_norm = "normalizedTriplets"

    client = OpenAI(
        api_key="LL-ZAmBAkmTiyeHR0zOm2zCBf5uT9mocUo9eCAcqQfFc4fE1GDtrEHJXTE8534AXXsl",
        base_url="https://api.llama-api.com"
    )

#https://www.llama-api.com/
    models = [
        # "llama-7b-32k", #trash
        "llama-13b-chat",  # god
        "llama-70b-chat",  # ok
        "mixtral-8x7b-instruct",  # ok
        "mistral-7b-instruct",  # ok
        # "mistral-7b", #trash
        # "NousResearch/Nous-Hermes-Llama2-13b", #trash
        # not working "falcon-7b-instruct",
        # not working "falcon-40b-instruct",
        # "alpaca-7b", #trash
        # "codellama-7b-instruct", #trash
        # "codellama-34b-instruct", #trash
        # "vicuna-7b",  # ok dai
        "vicuna-13b",  # top
    ]

    for model in models:
        print("Waiting for:", model)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=42,
            messages=[{
                "role": "user",
                "content": ("You will perform the open information extraction task. You will identify the named "
                            "entities in the content and then extract the relations between them. Use the same words."
                            "Based on the provided testimony, you will return triples which is formatted as <named "
                            "entity A, relation, named entity B>." + text + " The extracted triples formatted as <named entity A, "
                                                                            "relation, named entity B> are:")
            }]
        )

        # Define the filename based on the model name
        filename = f'extracted_text_from_{model.replace("/", "_")}_{file}.txt'

        # Save the response to a file
        save_text_to_file(response.choices[0].message.content, filename, output_directory_not_norm)
        kg.normalize_file(f"{output_directory_not_norm}/{filename}", f"{output_directory_norm}/_normalized{filename}")


def save_text_to_file(text, filename, output_directory):
    """Saves the provided text to a file, ensuring the directory exists."""
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create the full filepath
    filepath = os.path.join(output_directory, filename)

    # Open the file in append mode ('a') and write the text
    with open(filepath, 'a') as file:  # 'a' for append, 'w' for write (overwrite)
        file.write(text)

    # Print confirmation message
    print(f"Content saved to {filepath}")
