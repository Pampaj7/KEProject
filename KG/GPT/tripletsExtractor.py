import wikipedia
from openai import OpenAI
import re


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
    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    models = [
        "llama-13b-chat",
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
            temperature=0,
            seed=42,
            messages=[
                {"role": "user",
                 "content": "You will perform the open information extraction task. You will identify the named "
                            "entities in the content and then extract the relations between them."
                            "Based on the provided testimony, you will return triples which is formated as <named "
                            "entity A,"
                            "relation, named entity B>." + text + "The extracted triples formated as <named entity A, "
                                                                  "relation, named entity B> are:"
                 }

            ]

        )
        responses.append(response.choices[0].message.content)

    return responses, models


text_from_wiki = get_wikipedia_text()
text_from_AI = get_AI_response(text_from_wiki)

# Define the filename
text_from_AItxt = 'extracted_text_from_llama.txt'
# Open the file in write mode and write the triples


with open(text_from_AItxt, 'w') as file:
    file.write(text_from_AI[0][0])


