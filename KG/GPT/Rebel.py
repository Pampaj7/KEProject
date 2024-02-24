from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
}

# Text to extract triplets from
text = ("Deep learning is the subset of machine learning methods "
        "based on artificial neural networks (ANNs) with representation learning. The adjective "
        " refers to the use of multiple layers in the network. Methods used can be either supervised, "
        "semi-supervised or unsupervised.Deep-learning architectures such as deep neural networks, "
        "deep belief networks, recurrent neural networks, convolutional neural networks and transformers"
        " have been applied to fields including computer vision, speech recognition, natural language processing, "
        "machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection"
        "and board game programs, where they have produced results comparable to and in some cases surpassing human expert"
        " performance.Artificial neural networks were inspired by information processing and distributed communication "
        "nodes in biological systems. ANNs have various differences from biological brains. Specifically, artificial "
        "neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic "
        "(plastic) and analog. ANNs are generally seen as low quality models for brain function.")

# Tokenizer text
model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

# Generate
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs,
)

# Extract text
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

# Extract triplets
for idx, sentence in enumerate(decoded_preds):
    print(f'Prediction triplets sentence {idx}')
    print(extract_triplets(sentence))
