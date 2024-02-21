from openai import OpenAI
import wikiTextExtractor as we


def get_json_ld():
    text = (we.fetch_wikipedia_summary())

    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    response = client.chat.completions.create(
        model="llama-70b-chat",
        messages=[
            {"role": "user",
             "content": "**dont use a schema:name**,  **Build a knowledge Graph and give me the json-ld format from this text, I want something like this:{'@context': "
                        "{'schema': 'http://schema.org/'},'@graph': [{'@id': ''#algorithm','@type': 'schema:Thing','"
                        "schema:name': 'Algorithm'},:" + text}
        ]

    )
    return response.choices[0].message.content
