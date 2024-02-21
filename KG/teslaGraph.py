from transformers import pipeline

# Initialize the pipeline with an open-source GPT model, for example, GPT-Neo
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Example text to extract knowledge from
text = "Tesla, Inc. is an American electric vehicle and clean energy company based in Palo Alto, California. Founded in 2003 by engineers Martin Eberhard and Marc Tarpenning, it was later led by Elon Musk, who contributed more than $30 million to Tesla's new funding round as the Chairman of the Board."

# Generating text to extract entities and relationships
# We ask the model to generate text that lists entities and their relationships based on the input text
generated_text = generator(text, max_length=100, max_new_tokens=50)  # Assuming input is shorter than 50 tokens

# Print generated text
print(generated_text[0]['generated_text'])
