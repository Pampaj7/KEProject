# Import statements
import os
import openai
from wikipedia import wikipedia

from langchain.chat_models import ChatOpenAI

# Configuration settings
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


# Function definitions
def get_wikipedia_text(topic="deep learning"):
    """Fetches the summary of a given topic from Wikipedia."""
    search_results = wikipedia.search(topic)
    if not search_results:
        return "No results found for the topic."

    top_result_title = search_results[0]
    try:
        summary = wikipedia.summary(top_result_title)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple pages found for the topic. Consider being more specific. Suggestions: {e.options}"
    except wikipedia.exceptions.PageError:
        return "Page not found for the topic."


def ai_response(prompt):
    """Gets a response from an AI model for a given prompt."""
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)
    res = chat.predict(prompt)
    return res


# Main execution flow
if __name__ == "__main__":
    testimony = get_wikipedia_text()
    prompt_template = """
    You will perform the open information extraction task. You will identify the named entities in the content and then extract the relations between them.
    Based on the provided testimony, you will return triples which are formatted as <named entity A, relation, named entity B>.

    START of the testimony:
    {testimony}
    END of the testimony.

    The extracted triples formatted as <named entity A, relation, named entity B> are:
    """
    prompt = prompt_template.format(testimony=testimony)
    output_text = ai_response(prompt)

    text = "GPT-4model.txt"
    with open(text, "w") as file:
        file.write(output_text)
