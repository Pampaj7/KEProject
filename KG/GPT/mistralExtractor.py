from transformers import AutoModelForCausalLM, AutoTokenizer
import textSplitter as ts
import KG as kg

filename = "textSamples/marsDiary.csv"


def take_text(file):
    with open(file, 'r') as f:
        full_text = f.read()
    chunks = ts.split_text_into_chunks(full_text, chunk_size=1000)
    return chunks


def mistral_model(filename):
    device = "cuda"  # the device to load the model onto

    # DO NOT SET .TO(DEVICE) IT WILL CRASH
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_4bit=True,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # Set special tokens and pad token ID
    tokenizer.sep_token = "<SEP>"
    tokenizer.pad_token = "<PAD>"
    tokenizer.cls_token = "<CLS>"
    tokenizer.mask_token = "<MASK>"
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token ID to eos_token_id

    with open("tripletsTXTs/extracted_text_from_Mistral7B_CME_v1_LOCAL.txt", 'w') as triplet_file:
        for chunk in take_text(filename):
            messages = [
                {"role": "user",
                 "content": "You will perform the open information extraction task. You will identify the named "
                            "entities in the content and then extract the relations between them. Use the same words."
                            "Based on the provided testimony, you will return triples which is formatted as <named "
                            "entity A, relation, named entity B>." + chunk + " The extracted triples formatted as <named entity A, "
                                                                             "relation, named entity B> are:"
                 },

            ]

            # HERE SEEMS THAT .TO(DEVICE) MOTHING CHANGES
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            print(decoded)
            triplet_file.write(decoded[0] + "\n")
            kg.normalize_file("tripletsTXTs/extracted_text_from_Mistral7B_CME_v1_LOCAL.txt",
                              "normalizedTriplets/_normalizedextracted_text_from_Mistral7B_CME_v1_LOCAL.txt")


#mistral_model(filename)
