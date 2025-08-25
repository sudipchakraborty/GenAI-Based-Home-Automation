import os
from dotenv import load_dotenv
load_dotenv()  # load all the environment variables

# converting text to vector
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# # embeddings

# text = "This is a tutorial on OPENAI embedding"

# query_result = embeddings.embed_query(text)
# query_result






#####<OPENAI Embedding>########//////////////////////////////
# # converting text to vector
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# from langchain_openai import OpenAIEmbeddings
# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

# # embeddings

# text = "This is a tutorial on OPENAI embedding"

# query_result = embeddings.embed_query(text)
# print(query_result)

# ## Vector Embedding And Vector StoreDB
# from langchain_community.vectorstores import Chroma

# db = Chroma.from_documents(final_documents, embeddings_1024)

# ## serach from vector DB
# query = "It will be all the easier for us to conduct ourselves as belligerents"
# retrieved_results = db.similarity_search(query)
# print(retrieved_results)
##################################################################




#####Ollama Embedding/////////////////////////////////////////////
# from langchain_ollama import OllamaEmbeddings
# embeddings = (
#     OllamaEmbeddings(model="gemma:latest")  ## by default it uses llama2
# )
# r1 = embeddings.embed_documents(
#     [
#         "Alpha is the first letter of Greek alphabet",
#         "Beta is the second letter of Greek alphabet",
#     ]
# )
# kk=embeddings.embed_query("What is the second letter of Greek alphabet")
# print(kk)
#########################################################################



############<HuggingFace Embedding>/////////////////////////////////////
#  os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# text = "this is atest documents"
# query_result = embeddings.embed_query(text)
# print(query_result)
# doc_result = embeddings.embed_documents([text, "This is not a test document."])
# doc_result[0]
########################################################################