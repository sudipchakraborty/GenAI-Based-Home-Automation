


# #######<STEP-2>Text Splitter>###########################
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# final_documents = text_splitter.split_documents(docs)
# print(final_documents)


# from ContextLoader import context
# ctx=context.load_from_csv()
# print(ctx)     



from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("attention.pdf")
docs = loader.load()    

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20
)
split_docs=text_splitter.split_documents(docs)
print(split_docs)
 
#/////////////////////////////////////////////////////////



# Html text splitter/////////////////////////////////////
# from langchain_text_splitters import HTMLHeaderTextSplitter
 
# url = "https://www.drsudip.com"

# headers_to_split_on = [
#     ("h1", "Header 1"),
#     ("h2", "Header 2"),
#     ("h3", "Header 3"),
#     ("h4", "Header 4"),
# ]

# html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
# html_header_splits = html_splitter.split_text_from_url(url)
# print(html_header_splits)
#/////////////////////////////////////////////////////////




#//// JSON text split//////////////////////////////////////
import json
import requests
from langchain_text_splitters import RecursiveJsonSplitter

json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = json_splitter.split_json(json_data)
# print(json_chunks)
## The splitter can also output documents
docs = json_splitter.create_documents(texts=[json_data])

# for doc in docs[:3]:
#     print(doc)

texts = json_splitter.split_text(json_data)

print(texts[0])
print(texts[1])









