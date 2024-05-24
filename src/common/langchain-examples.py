from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.helper import load_pdf, text_split, download_hugging_face_embeddings, llm_factory
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def example1():
    '''Simple chain'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}")
    ])

    llm=llm_factory("llama2-chat")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"input": "how can langsmith help with testing?"})
    print(answer)

def example2():
    '''Retrieval chain'''
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
                                              """)

    # model

    llm = llm_factory("llama2-chat")

    # chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # retriever
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "what is this document about ?"})
    print(response["answer"])

example2()
