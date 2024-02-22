from openai import OpenAI
import wikiTextExtractor as we


def get_json_ld():
    text = (we.fetch_wikipedia_summary())

    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    models = ["llama-7b-chat", "llama-7b-32k", "llama-13b-chat", "llama-70b-chat", "mixtral-8x7b-instruct",
              "mistral-7b-instruct", "mistral-7b", "NousResearch/Nous-Hermes-Llama2-13b", "falcon-7b-instruct",
              "falcon-40b-instruct", "alpaca-7b", "codellama-7b-instruct", "codellama-13b-instruct",
              "codellama-34b-instruct", "openassistant-llama2-70b", "vicuna-7b", "vicuna-13b", "vicuna-13b-16k"]

    responses = []
    for i in models:
        response = client.chat.completions.create(
            model=i,
            messages=[
                {"role": "user",
                 "content": "**dont use a schema:name**,  **Build a knowledge Graph and give me the json-ld format from this text, I want something like this:{'@context': "
                            "{'schema': 'http://schema.org/'},'@graph': [{'@id': ''#algorithm','@type': 'schema:Thing','"
                            "schema:name': 'Algorithm'},:" + text}
            ]

        )
        responses.append(response)

    return responses, models
