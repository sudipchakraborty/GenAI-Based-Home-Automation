# contentLoader.py
# Unified document loader utilities for multiple sources.
# Works with: TXT, PDF, CSV, Web URLs, ArXiv, Wikipedia
# Optional: text chunking via RecursiveCharacterTextSplitter

from __future__ import annotations
from typing import Iterable, List, Optional, Union

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Lazy imports (only if you use them)
try:
    from langchain_community.document_loaders import ArxivLoader
except Exception:
    ArxivLoader = None

try:
    from langchain_community.document_loaders import WikipediaLoader
except Exception:
    WikipediaLoader = None

import bs4
import os


# ----------------------------
# Core chunking helper
# ----------------------------
def chunk_docs(
    docs,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    add_start_index: bool = True,
):
    """
    Split Documents into chunks for downstream embedding/RAG.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    return splitter.split_documents(docs)


# ----------------------------
# File-based loaders
# ----------------------------
def load_from_text(path: str, encoding: str = "utf-8", autodetect_encoding: bool = True):
    """
    Load content from a plain text file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")

    loader = TextLoader(path, encoding=encoding, autodetect_encoding=autodetect_encoding)
    return loader.load()


def load_from_pdf(path: str):
    """
    Load content from a PDF file using PyPDFLoader.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(path)
    return loader.load()


def load_from_csv(path: str, csv_args: Optional[dict] = None):
    """
    Load content from a CSV file as Documents (each row -> one Document).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    loader = CSVLoader(file_path=path, csv_args=csv_args or {})
    return loader.load()


# ----------------------------
# Web / API loaders
# ----------------------------
def load_from_web(
    urls: Union[str, Iterable[str]],
    css_classes: Iterable[str] = ("post-title", "post-content", "post-header"),
):
    """
    Load content from one or many web pages.
    """
    if isinstance(urls, str):
        urls = (urls,)

    loader = WebBaseLoader(
        web_paths=tuple(urls),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=tuple(css_classes))),
    )
    return loader.load()


def load_from_arxiv(query: str, load_max_docs: int = 2):
    """
    Load papers from ArXiv by query or ID (e.g., '1706.03762').
    """
    if ArxivLoader is None:
        raise ImportError("ArxivLoader not available. Install langchain_community extras if needed.")

    loader = ArxivLoader(query=query, load_max_docs=load_max_docs)
    return loader.load()


def load_from_wikipedia(query: str, load_max_docs: int = 2, lang: str = "en"):
    """
    Load pages from Wikipedia.
    """
    if WikipediaLoader is None:
        raise ImportError("WikipediaLoader not available. Install langchain_community extras if needed.")

    loader = WikipediaLoader(query=query, load_max_docs=load_max_docs, lang=lang)
    return loader.load()


# ----------------------------
# Auto-dispatch by source type
# ----------------------------
def load_documents(
    source: Union[str, Iterable[str]],
    source_type: Optional[str] = None,
    chunk: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    **kwargs,
):
    """
    Unified entry point.

    Parameters
    ----------
    source : str | Iterable[str]
        - File path (TXT/PDF/CSV), URL, ArXiv/Wikipedia query
        - Or iterable of URLs for web loading
    source_type : Optional[str]
        One of {"text","pdf","csv","web","arxiv","wikipedia"}.
        If None, will infer from file extension for file paths.
    chunk : bool
        Whether to split documents into chunks.
    chunk_size : int
    chunk_overlap : int
    kwargs : dict
        Extra args passed to specific loaders (e.g., csv_args for CSV, lang for Wikipedia)

    Returns
    -------
    list[Document] | list[Chunked Document]
    """

    if source_type is None and isinstance(source, str):
        # try to infer from extension
        ext = os.path.splitext(source)[1].lower()
        if ext in {".txt", ".md", ".rst"}:
            source_type = "text"
        elif ext in {".pdf"}:
            source_type = "pdf"
        elif ext in {".csv"}:
            source_type = "csv"

    if source_type == "text":
        docs = load_from_text(source, **{k: v for k, v in kwargs.items() if k in {"encoding", "autodetect_encoding"}})
    elif source_type == "pdf":
        docs = load_from_pdf(source)
    elif source_type == "csv":
        docs = load_from_csv(source, csv_args=kwargs.get("csv_args"))
    elif source_type == "web":
        docs = load_from_web(source, css_classes=kwargs.get("css_classes", ("post-title", "post-content", "post-header")))
    elif source_type == "arxiv":
        docs = load_from_arxiv(query=str(source), load_max_docs=int(kwargs.get("load_max_docs", 2)))
    elif source_type == "wikipedia":
        docs = load_from_wikipedia(query=str(source), load_max_docs=int(kwargs.get("load_max_docs", 2)), lang=kwargs.get("lang", "en"))
    else:
        raise ValueError(
            "Unable to determine source_type. "
            "Pass source_type explicitly: one of {'text','pdf','csv','web','arxiv','wikipedia'} "
            "or provide a file path with a known extension."
        )

    if chunk:
        return chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return docs


# ----------------------------
# Quick manual tests
# ----------------------------
if __name__ == "__main__":
    # Examples (uncomment what you want to test):

    # TXT
    # print(load_documents("Attendance.txt", source_type="text")[:1])

    # PDF
    # print(load_documents("NIPS-2017-attention-is-all-you-need-Paper.pdf", source_type="pdf")[:1])

    # CSV
    # print(load_documents("AUG 2025.csv", source_type="csv")[:1])

    # Web
    # print(load_documents(
    #     ["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    #     source_type="web",
    # )[:1])

    # ArXiv
    # print(load_documents("1706.03762", source_type="arxiv")[:1])

    # Wikipedia
    # print(load_documents("Generative AI", source_type="wikipedia", load_max_docs=1)[:1])

    pass
