# #######
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import bs4
# ########

# ####(1)#############################################################
# # Read from text file
# # loader = TextLoader("Attendance.txt")
# # documents = loader.load()
# # print(documents)

# # Read from pdf file
# loader = PyPDFLoader('NIPS-2017-attention-is-all-you-need-Paper.pdf')
# docs = loader.load()
# # print(docs)

# ## Web based loader ##################################################
# # loader = WebBaseLoader(
# #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
# #     bs_kwargs=dict(
# #         parse_only=bs4.SoupStrainer(
# #             class_=("post-title", "post-content", "post-header")
# #         )
# #     )
# # )
# # wl_content=loader.load()
# # print(wl_content)
# #----------------------------------------------------------------------

# ## Arxiv
# # from langchain_community.document_loaders import ArxivLoader
# # docs = ArxivLoader(query="1706.03762", load_max_docs=2).load()
# # print(docs)
# #------------------------------------------------
#___________________________________________________________________________________________
# # from eikipedia loader
# # from langchain_community.document_loaders import WikipediaLoader
# # docs = WikipediaLoader(query="Generative AI", load_max_docs=2).load()
# # len(docs)
# # print(docs)

#___________________________________________________________________________________________
def load_from_csv():
    """
        @brief load content from .csv file
        @param x The first number.
        @param y The second number.
        @return The product of x and y.
    """
    loader = CSVLoader(file_path="AUG 2025.csv")
    docs = loader.load()
    return docs
#___________________________________________________________________________________________

   
 