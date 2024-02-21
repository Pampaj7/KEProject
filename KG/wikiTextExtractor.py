import wikipedia


def fetch_wikipedia_summary():
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

