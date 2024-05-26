from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.helper import load_pdf, text_split, download_hugging_face_embeddings, llm_factory

from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def example1():
    '''Prompt template'''

    prompt_template = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}."
    )

    prompt = prompt_template.format(adjective="funny", content="chicken")

    llm=llm_factory("GPT-4")
    answer = llm.invoke(prompt)

    return answer

logging.info(example1())
