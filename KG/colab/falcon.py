import torch
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

STYLE = "<|prompt|>{instruction}<|endoftext|><|answer|>"


def ai_response(prompt):
    res = generate_text(
        prompt,
        min_new_tokens=2,
        max_new_tokens=1024,
        do_sample=True,
        num_beams=1,
        temperature=float(0.3),
        repetition_penalty=float(1.2),
        renormalize_logits=True
    )
    return res[0]["generated_text"]

class H2OTextGenerationPipeline(TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = STYLE

    def preprocess(
        self, prompt_text, prefix="", handle_long_generation=None, **generate_kwargs
    ):
        prompt_text = self.prompt.format(instruction=prompt_text)
        return super().preprocess(
            prompt_text,
            prefix=prefix,
            handle_long_generation=handle_long_generation,
            **generate_kwargs,
        )

    def postprocess(
        self,
        model_outputs,
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
    ):
        records = super().postprocess(
            model_outputs,
            return_type=return_type,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for rec in records:
            rec["generated_text"] = (
                rec["generated_text"]
                .split("<|answer|>")[1]
                .strip()
                .split("<|prompt|>")[0]
                .strip()
            )
        return records



tokenizer = AutoTokenizer.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    use_fast=False,
    padding_side="left",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

prompt_qa = """ You are a helpful AI. Provided some testimony from Human, you will extract pairs of question and answer from it.

START of the testimony:
XXX
END of the testimony.

The extracted question and answer pairs from the testimony are:
"""

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