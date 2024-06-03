from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from github import fetch_github_issues
from note import note_tool

load_dotenv()

# Connect to vectorstore
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "github-agent"
embeddings = OpenAIEmbeddings()

def connect_to_vectorstore():
    owner = "techwithtim"
    repo = "Flask-Web-App-Tutorial"
    issues = fetch_github_issues(owner, repo)

    vectorstore_from_docs = PineconeVectorStore.from_documents(
        issues,
        index_name=index_name,
        embedding=embeddings
    )

    print(vectorstore_from_docs)

# verify if vectors are already loaded in the index before connecting to vectorstore
def load_vectors():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    if index.describe_index_stats().total_vector_count == 0:
        connect_to_vectorstore()
    else:
        print("Vectors already loaded in index")

load_vectors()

# Loading index
vstore=PineconeVectorStore.from_existing_index(index_name, embeddings)
print(vstore)

# CREATE AGENT
# set-up tool
retriever = vstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "github_search",
    "Search for information about github issues. For any questions about github issues, you must use this tool!"
)

# download prompt from langchain hub: prompt that tells agent how it should behave and it should utilise the tools we give to it
prompt = hub.pull("hwchase17/openai-functions-agent")

# agent
llm = ChatOpenAI()
tools = [retriever_tool, note_tool]
agent = create_tool_calling_agent(llm, tools, prompt)

# execute agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#loop to continuously ask the agent questions
while (question := input("Ask a question about Github issues (q to quit): "))!= "q":
    result = agent_executor.invoke({"input":question})
    print(result["output"])
