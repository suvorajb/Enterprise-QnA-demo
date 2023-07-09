import langchain, pinecone

from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


directory_path = 'data'

PINECONE_ENV = "asia-northeast1-gcp"
PINECONE_API_KEY = "43886c44-bcb6-4599-ae9e-7e41236b997b"
PINECONE_INDEX_NAME = "qna-private-enterprise-business-data-pinecone"

OPEMAI_API_KEY = "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"

# Set up Pinecone client
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)
print("******** Pinecone initialized. Index status ********\n")
print(str(index.describe_index_stats()))
print("***************************************************\n")

# Load the source documents (e.g. frequently asked Q/A for ecommerce site)
def load_documents(directory_path):
  print("\nSTEP 1:: Scanning directory & loading all the documents ")
  loader = DirectoryLoader(directory_path)
  documents = loader.load()
  print("Found ecommerce FAQ sample ... loading the document for chunking\n")
  return documents


# split or chunk the texts based on fixed chunk size (1000)
def split_docs(documents, chunk_size=500, chunk_overlap=20):
  print("\nSTEP 2::  Started Chunking the document ")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  chunks = text_splitter.split_documents(documents)
  print("***** Total Number of documents chunked:: " + str(len(chunks)) + "\n")
  return chunks


# Generate Embeddings using OpenAI's Embeddings model and store into Pinecone database
def generate_embeddings():
  print("\nSTEP 3::  Initializing OpenAI Embeddings model for converting docs into vectors")
  embeddings = OpenAIEmbeddings(openai_api_key=OPEMAI_API_KEY, model="text-embedding-ada-002")
  return embeddings


def store_embeddings_in_pinecone(embeddings):
  print("\nSTEP 4::  Store the embeddings into the Pinecone vector db ")
  index = Pinecone.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
  return index

# Retrieve similar documents from Pinecone
def get_similiar_docs(query, k=1):
  similar_docs = index.similarity_search(query, k=k)
  return similar_docs


def get_answer(query):
  model_name = "text-davinci-003"
  llm = OpenAI(model_name=model_name,  temperature=0, openai_api_key=OPEMAI_API_KEY)
  chain = load_qa_chain(llm, chain_type="stuff")

  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return query + " \nAnswer:: " + answer + "\n\n"

print("\n****** Starting Enterprise data ingestion to load FAQ articles into Pinecone vector db******")
loaded_docs = load_documents(directory_path)
chunks  = split_docs(loaded_docs)

embeddings = generate_embeddings()
index = store_embeddings_in_pinecone(embeddings)
print("*************** Ingestion completed ***************\n")
print("***************************************************\n")


print("\n****** Starting Retrieval using OpenAI and context from Pinecone vectordb ******\n")

query1 = "How to cancel my order ?"
print("Question 1:: " + get_answer(query1))

query2 = "Do you have any loyalty program ?"
print("Question 2:: " + get_answer(query2))

query3 = "How do I reset my password ?"
print("Question 3:: " + get_answer(query3))

query4 = "Can I return a product purchased using store credit?"
print("Question 4:: " + get_answer(query4))

query5 = "What to do if a wrong item is received ?"
print("Question 5:: " + get_answer(query5))