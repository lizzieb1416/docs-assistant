from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.helper import load_pdf, text_split, download_hugging_face_embeddings, llm_factory
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.prompt import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_tree(starting_directory):
    tree = ""
    for root, directories, files in os.walk(starting_directory):
        tree += f"Directory: {root}\n"

    return tree

def doc_name_finder():
    output_parser = StrOutputParser()

    # extract data and make chunks
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)

    # Download embedding model
    embeddings = download_hugging_face_embeddings()
    vector = FAISS.from_documents(text_chunks, embeddings)

    # prompt
    prompt = ChatPromptTemplate.from_template("""You are an specialist in analysing documents.
                                                Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # model

    llm = llm_factory("GPT-4")

    # chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # retriever
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": "propose a name for the file and its corresponding date (if it exists) with the following format: '<year-month-day> <name of the file>'"
        })

    return response["answer"]

def directory_finder(directory_tree):
    file_name = doc_name_finder()
    prompt = PromptTemplate(
        input_variables=["directory_tree", "file_name"],
        template="Question: According to this directory tree {directory_tree}, where should this file {file_name} file be placed?. Choose only one folder. The answser returned must be only the path of the folder, nothing else, no explanatinos needed"
    )

    llm=llm_factory("GPT-4")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"input": prompt, "directory_tree": generate_tree(directory_tree), "file_name": file_name})
    print(answer)



print(doc_name_finder())
print(directory_finder("my_directory"))



# TODO: ne pas utiliset les functinalités de chat, mais plutôt le text generation
# TODO: apprendre à mieux utiliser les prompts
# TODO: la function generate_tree doit être exécuté une seule fois, pas à chaque fois que directory_finder est appelé
