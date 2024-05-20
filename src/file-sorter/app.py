from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

# extract data and make chunks
extracted_data = load_pdf("data")
#print("Extracted data:", extracted_data)

text_chunks = text_split(extracted_data)
#print("length of text chunks: ", len(text_chunks))

# Download embedding model
embeddings = download_hugging_face_embeddings()
docsearch = FAISS.from_documents(text_chunks, embeddings)

# prompt
query = "propose a name for the file and its corresponding date (if it exists) with the following format: '<year-month-day> <name of the file>'"

# load the model
# TODO: try with model that generates text and not qa
llm=CTransformers(model="../../models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

#print(llm)
