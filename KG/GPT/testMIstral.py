from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import os

model_name_or_path = "TheBloke/Mistral-7B-v0.1-AWQ"
filename = "testi/Mistral7B_CME_v1.csv"

# Set the device dynamically
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, safetensors=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

with open(filename, 'r') as f:
    text = f.read()

prompt = (f"You will perform the open information extraction task. You will identify the named "
          f"entities in the content and then extract the relations between them. Use the same words. "
          f"Based on the provided testimony, you will return triples which is formatted as <named "
          f"entity A, relation, named entity B>. {text} The extracted triples formatted as <named entity A, ")

tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

# Generate output
generation_output = model.generate(
    tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=32,
    use_cache=True,
)

print("Output: ", tokenizer.decode(generation_output[0]))
