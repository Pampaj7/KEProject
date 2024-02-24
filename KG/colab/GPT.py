import openai
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


def ai_response(prompt):
    res = chat.predict(prompt)
    return res


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
