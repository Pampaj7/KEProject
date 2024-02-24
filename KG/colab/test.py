import openai
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
openai.api_key = "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"