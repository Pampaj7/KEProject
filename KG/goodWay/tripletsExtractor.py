import wikipedia
from openai import OpenAI
import re


def get_wikipedia_text():
    # Search for the topic on Wikipedia
    topic = "deep learning"
    search_results = wikipedia.search(topic)

    if not search_results:
        return "No results found for the topic."

    # Get the top search result's title
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


def get_AI_response(text = get_wikipedia_text()):

    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    models = [
        "llama-7b-chat",
        # "llama-7b-32k", "llama-13b-chat", "llama-70b-chat", "mixtral-8x7b-instruct",
        # "mistral-7b-instruct", "mistral-7b", "NousResearch/Nous-Hermes-Llama2-13b", "falcon-7b-instruct",
        # "falcon-40b-instruct", "alpaca-7b", "codellama-7b-instruct", "codellama-34b-instruct", "vicuna-7b",
        # "vicuna-13b",
    ]

    responses = []
    for i in models:
        print("Waiting for:", i)
        response = client.chat.completions.create(
            model=i,
            messages=[
                {"role": "user",
                 "content": "Please analyze the provided text and extract key information "
                            "in the form of triplets, consisting of subject, predicate, and object, "
                            "to help construct a Knowledge Graph. Ensure to identify main concepts, "
                            "their attributes, and relationships between them. Format your response without useless text"+ text}

            ]

        )
        responses.append(response.choices[0].message.content)

    return responses, models


def write_triplets_to_file(triplets_text, filename="triplets.txt"):
    # Split the text into lines
    lines = triplets_text.strip().split('\n')

    # Open a file to write the triplets
    with open(filename, 'w', encoding='utf-8') as file:
        for line in lines:
            # Assuming each line is a triplet formatted as "Subject: ..., Predicate: ..., Object: ..."
            parts = line.split('\n')
            # Reformat each part to extract the specific information after the colon
            for part in parts:
                subject = part.split("Subject: ")[1].split("\n")[0] if "Subject: " in part else ""
                predicate = part.split("Predicate: ")[1].split("\n")[0] if "Predicate: " in part else ""
                object = part.split("Object: ")[1].split("\n")[0] if "Object: " in part else ""
                # Write the formatted triplet to the file
                file.write(f"{subject} - {predicate} - {object}\n")


print(get_wikipedia_text())
