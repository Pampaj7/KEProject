from openai import OpenAI


def get_json_ld():
    text = ("In mathematics and computer science, an algorithm is a finite sequence of rigorous instructions, "
            "typically used to solve a class of specific problems or to perform a computation.[1] Algorithms "
            "are used as specifications for performing calculations and data processing. More advanced algorithms "
            "can use conditionals to divert the code execution through various routes (referred to as automated decision-making) "
            "and deduce valid inferences (referred to as automated reasoning), achieving automation eventually. Using human "
            "characteristics as descriptors of machines in metaphorical ways was already practiced by Alan Turing with terms "
            "such as 'memory', 'search' and 'stimulus'.[2]In contrast, a heuristic is an approach to problem solving that may "
            "not be fully specified or may not guarantee correct or optimal results, especially in problem domains where there "
            "is no well-defined correct or optimal result.[3] For example, social media recommender systems rely on heuristics "
            "in such a way that, although widely characterized as 'algorithms' in 21st century popular media, cannot deliver correct "
            "results due to the nature of the problem.As an effective method, an algorithm can be expressed within a finite "
            "amount of space and time[4] and in a well-defined formal language[5] for calculating a function.[6] Starting from "
            "an initial state and initial input (perhaps empty),[7] the instructions describe a computation that, when executed, "
            "proceeds through a finite[8] number of well-defined successive states, eventually producing 'output'[9] and terminating "
            "at a final ending state. The transition from one state to the next is not necessarily deterministic; some algorithms, "
            "known as randomized algorithms, incorporate random input.[10]")

    client = OpenAI(
        api_key="LL-5OPT3xLPTWxSff1zUH42hQxtOBQPTLhqOlnHCMLrDYpRGIwgXaum7Rj8LWu31eXV",
        base_url="https://api.llama-api.com"
    )

    response = client.chat.completions.create(
        model="llama-70b-chat",
        messages=[
            {"role": "user",
             "content": "****,  **Build a knowledge Graph and give me the json-ld format from this text, I want something like this:{'@context': "
                        "{'schema': 'http://schema.org/'},'@graph': [{'@id': ''#algorithm','@type': 'schema:Thing','"
                        "schema:name': 'Algorithm','schema:description': 'A finite sequence.'},:" + text}
        ]

    )
    return response.choices[0].message.content


