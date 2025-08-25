
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter



####<FAISS DB>///////////////////////////////////////////////////
# loader = TextLoader("Attendance.txt")
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
# docs = text_splitter.split_documents(documents)
# # print(docs)

# embeddings = OllamaEmbeddings()
# db = FAISS.from_documents(docs, embeddings)
# print(db)

# ### querying 
# query="How does the speaker describe the desired outcome of the war?"
# docs=db.similarity_search(query)
# docs[0].page_content


# retriever=db.as_retriever()
# docs=retriever.invoke(query)
# docs[0].page_content

# docs_and_score=db.similarity_search_with_score(query)
# docs_and_score

# embedding_vector=embeddings.embed_query(query)
# embedding_vector

# docs_score=db.similarity_search_by_vector(embedding_vector)
# docs_score

# ### Saving And Loading
# db.save_local("faiss_index")

# new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
# docs=new_db.similarity_search(query)

# docs
#////////////////////////////////////////////////////////////////////////



#####ChromaDB////////////////////////////////////////////////////////////
## building a sample vectordb
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = TextLoader("speech.txt")
# data = loader.load()
# data

# # Split
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# splits = text_splitter.split_documents(data)

# embedding=OllamaEmbeddings()
# vectordb=Chroma.from_documents(documents=splits,embedding=embedding)
# vectordb

# ## query it
# query = "What does the speaker believe is the main reason the United States should enter the war?"
# docs = vectordb.similarity_search(query)
# docs[0].page_content

# ## Saving to the disk
# vectordb=Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db")

# # load from disk
# db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
# docs=db2.similarity_search(query)
# print(docs[0].page_content)

# ## similarity Search With Score
# docs = vectordb.similarity_search_with_score(query)
# docs

# ### Retriever option
# retriever=vectordb.as_retriever()
# retriever.invoke(query)[0].page_content
#####################################################################