from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

# Define the BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable loading the model in 4-bit
    bnb_4bit_use_double_quant=True,  # Use double quantization
    bnb_4bit_quant_type="nf4",  # Set the quantization type to 'nf4'
    bnb_4bit_compute_dtype="bfloat16"  # Use bfloat16 for computation
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

"""
model_do = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype="auto",  # Use automatic data type detection for PyTorch
    low_cpu_mem_usage=True,  # Optimize for lower CPU memory usage
    device_map="auto",  # Automatically distribute the model across available devices
    bitsandbytes_config=bnb_config  # Apply the BitsAndBytes configuration
)
"""


def ai_response(prompt):
    """
    res = model_do(prompt,
                   do_sample=True,
                   top_k=10,
                   num_return_sequences=1,
                   eos_token_id=tokenizer.eos_token_id,
                   max_length=800, )
"""

    res = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=800,
    )

    return res[0]["generated_text"].replace(prompt, "")


ai_response(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n')

testimony = """

Q1. Please state your full name, title, business address and by whom you are
employed?
My name is Marisa Ayala. My business address is One Energy Plaza, Detroit,
Michigan 48226. I am employed by DTE Gas Company as Manager – Gas
Operations

Q2. Whatis youreducational background?
I graduated from Michigan State University, Lansing in 2004 with a Bachelor of
Science degree in Engineering and a Cognate in Business and Supply Chain
 Management. I earned my Master of Business Administration degree from the
 UniversityofMichigan, AnnArborin2015

 Q3. Whatis yourwork experience?
 I worked for Shertrack, a software startup from 2005 through 2006 as a Data
 Analyst/ConsultingEngineerwhereIanalyzedmodelsoforderingpatterns,safety
 stock,productionandmaterials planning.
 In 2006 I joined Federal Mogul Corporation as an Inventory Analyst and was
 promoted to Supply Chain Team Leader in 2008. In this capacity I analyzed all
 aspects of the supply chain process across the organization. I evaluated
 procurement,supplierandinventorysystems,orderingfrequency,safetystockand
 shipping. I designed supply chain strategy and presented creative supplier and
 internal plants recoveryplanstreamliningsupplychainprocesses.
 In2010, IjoinedBehr-HellaThermocontrols as aLogistics lead.Iwas responsible
 foroptimizingthematerials management andproductionplanningprocess.

"""

prompt_qa = """ You are a helpful AI. Provided some testimony from Human, you will extract pairs of question and answer from it.

START of the testimony:
XXX
END of the testimony.

The extracted question and answer pairs from the testimony are:
"""

prompt = prompt_qa.replace("XXX", testimony)
print(ai_response(prompt))

prompt_qa_json = """ You are a helpful AI. Provided some testimony from Human, you will extract pairs of question and answer from it.

START of the testimony:
XXX
END of the testimony.

The extracted question and answer pairs in JSON from the testimony are:
"""

prompt = prompt_qa_json.replace("XXX", testimony)
print(ai_response(prompt))

prompt_summary = """ You are a helpful AI. Provided some testimony from Human, you will summary important information from it.

START of the testimony:
XXX
END of the testimony.

The summarization is in sentences:
"""

prompt = prompt_summary.replace("XXX", testimony)
print(ai_response(prompt))

prompt_triple = """ You will perform the open information extraction task. You will identify the named entities in the content and then extract the relations between them.
Based on the provided testimony, you will return triples which is formated as [named entity A, relation, named entity B].

START of the testimony:
XXX
END of the testimony.

The extracted triples formated as [named entity A, relation, named entity B] are:
"""

prompt = prompt_triple.replace("XXX", testimony)
print(ai_response(prompt))
