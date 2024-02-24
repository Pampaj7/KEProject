from openai import OpenAI
import wikiTextExtractor as we


def get_json_ld():
    text = (we.fetch_wikipedia_summary())
    text = ""

    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    models = [
        "llama-7b-chat",
        #"llama-7b-32k", "llama-13b-chat", "llama-70b-chat", "mixtral-8x7b-instruct",
        #"mistral-7b-instruct", "mistral-7b", "NousResearch/Nous-Hermes-Llama2-13b", "falcon-7b-instruct",
        #"falcon-40b-instruct", "alpaca-7b", "codellama-7b-instruct", "codellama-34b-instruct", "vicuna-7b",
        #"vicuna-13b",
    ]

    responses = []
    for i in models:
        print("Waiting for:", i)
        response = client.chat.completions.create(
            model=i,
            messages=[
                {"role": "user",
                 "content": "Extract triplets like subject-predicate-object from this text: " + text}

            ]

        )
        responses.append(response.choices[0].message.content)

    return responses, models
