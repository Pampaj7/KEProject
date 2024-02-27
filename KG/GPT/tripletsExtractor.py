import wikipedia
from openai import OpenAI
import os


def get_wikipedia_text():
    # Search for the topic on Wikipedia
    topic = "deep learning"
    search_results = wikipedia.search(topic)

    if not search_results:
        return "No results found for the topic."

    # Get the top search result's titlepip
    top_result_title = search_results[0]

    try:
        # Get the summary of the top result
        summary = wikipedia.summary(top_result_title)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        return f"Multiple pages found for the topic. Consider being more specific. Suggestions: {e.options}"
    except wikipedia.exceptions.PageError:
        # Handle pages not found
        return "Page not found for the topic."


def get_AI_response(text=get_wikipedia_text()):
    output_directory = "tripletsTXTs"
    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    models = [
        "llama-13b-chat",  # ok
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
        "vicuna-7b",  # ok dai
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
                            "entities in the content and then extract the relations between them. "
                            "Based on the provided testimony, you will return triples which is formatted as <named "
                            "entity A, relation, named entity B>." + text + " The extracted triples formatted as <named entity A, "
                                                                            "relation, named entity B> are:")
            }]
        )

        # Define the filename based on the model name
        filename = f'extracted_text_from_{model.replace("/", "_")}.txt'

        filepath = os.path.join(output_directory, filename)
        # Open the file in write mode and write the triples
        with open(filepath, 'w') as file:
            file.write(response.choices[0].message.content)
        print(f"Output saved to {filepath}")


text_from_wiki = get_wikipedia_text()
get_AI_response(text_from_wiki)
