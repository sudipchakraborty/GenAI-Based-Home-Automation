# textSplitter/splitters.py
"""
Reusable text/document splitting utilities built on langchain_text_splitters.
Supports:
- Recursive character splits (generic documents)
- Character splits (raw text)
- Token-based splits (by token count)
- Markdown header splits
- Code-aware splits (language heuristics)
- HTML header splits (from URL or raw HTML)
- JSON recursive splits (from URL or in-memory JSON)
- PDF -> Documents -> split

All functions return a list of `Document` objects unless noted.
"""

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Optional, Union
from dataclasses import dataclass

# LangChain splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
)
# Code-aware splitter helpers
from langchain_text_splitters import Language

# Loaders (used by helpers)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import Document

# Optional deps
import bs4  # noqa: F401  (kept if you later add HTML parsing from raw HTML)
import os
import json
from typing import Any

# ------------------------------
# Generic configs
# ------------------------------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

@dataclass
class SplitConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    add_start_index: bool = True


# ------------------------------
# Core splitters
# ------------------------------
def split_documents_recursive(
    docs: Sequence[Document],
    cfg: SplitConfig = SplitConfig(),
) -> List[Document]:
    """Split LangChain Documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=cfg.add_start_index,
    )
    return splitter.split_documents(docs)


def split_text_character(
    text: str,
    separator: str = "\n\n",
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> List[Document]:
    """Split raw text using CharacterTextSplitter -> Documents."""
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.create_documents([text])


def split_text_by_tokens(
    text: str,
    tokens_per_chunk: int = 256,
    tokens_overlap: int = 32,
    encoding_name: Optional[str] = None,  # if None, langchain chooses default
) -> List[Document]:
    """Split text by approximate token count."""
    splitter = TokenTextSplitter(
        chunk_size=tokens_per_chunk,
        chunk_overlap=tokens_overlap,
        encoding_name=encoding_name,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=ch, metadata={"splitter": "token"}) for ch in chunks]


def split_markdown(
    text: str,
    headers_to_split_on: Sequence[Tuple[str, str]] = (
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ),
) -> List[Document]:
    """Split Markdown by headers into Documents preserving section metadata."""
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return md_splitter.split_text(text)


def split_code(
    text: str,
    language: Language = Language.PYTHON,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
) -> List[Document]:
    """
    Code-aware recursive split. Choose Language via:
      Language.PYTHON, Language.JS, Language.CPP, Language.GO, Language.JAVA, etc.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.create_documents([text])


# ------------------------------
# PDF helpers
# ------------------------------
def split_pdf(
    path: str,
    cfg: SplitConfig = SplitConfig(),
) -> List[Document]:
    """Load a PDF into Documents then split recursively."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    docs = PyPDFLoader(path).load()
    return split_documents_recursive(docs, cfg)


# ------------------------------
# HTML header splits (URL)
# ------------------------------
def split_html_from_url(
    url: str,
    headers_to_split_on: Sequence[Tuple[str, str]] = (
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ),
) -> List[Document]:
    """
    Fetch HTML from a URL and split by header tags.
    Note: You need to fetch HTML yourself or use a loader.
    This function is kept minimal and expects you to provide the HTML string
    if you want to split offline. For web loading, use your own loader first.
    """
    import requests
    html = requests.get(url, timeout=20).text  # simple fetch
    splitter = MarkdownHeaderTextSplitter.from_html(
        headers_to_split_on=headers_to_split_on
    )
    return splitter.split_text(html)


# ------------------------------
# JSON recursive splits
# ------------------------------
def split_json_obj(
    data: Any,
    max_chunk_size: int = 300,
) -> List[Document]:
    """Split an in-memory JSON-like object into Documents."""
    splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
    return splitter.create_documents(texts=[data])


def split_json_from_url(
    url: str,
    max_chunk_size: int = 300,
    timeout: int = 20,
) -> List[Document]:
    """Fetch JSON from URL and split into Documents."""
    import requests
    json_data = requests.get(url, timeout=timeout).json()
    return split_json_obj(json_data, max_chunk_size=max_chunk_size)


# ------------------------------
# Convenience: generic dispatcher
# ------------------------------
def split_auto(
    source: Union[str, Sequence[Document]],
    cfg: SplitConfig = SplitConfig(),
    *,
    assume_language: Optional[Language] = None,
) -> List[Document]:
    """
    Convenience router:
      - If `source` is a path to .pdf: load & split PDF.
      - If `source` is raw text: recursive split (uses cfg).
      - If `source` is a list of Documents: recursive split.
    """
    if isinstance(source, list) and source and isinstance(source[0], Document):
        return split_documents_recursive(source, cfg)

    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".pdf":
            return split_pdf(source, cfg)
        # treat as raw text otherwise
        if assume_language:
            return split_code(source, language=assume_language,
                              chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        return split_text_character(source, separator="\n\n",
                                    chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    raise ValueError("Unsupported input for split_auto. Pass a PDF path, raw text, or a list of Documents.")


###################################
# 1) Split a PDF into chunks
# from textSplitter import split_pdf, SplitConfig

# chunks = split_pdf("attention.pdf", SplitConfig(chunk_size=500, chunk_overlap=50))
# print(len(chunks))

# 2) Split raw text by characters
# from textSplitter import split_text_character
# chunks = split_text_character("line1\n\nline2\n\nline3", chunk_size=10, chunk_overlap=2)

# 3) Token-based split
# from textSplitter import split_text_by_tokens
# chunks = split_text_by_tokens("Some long text ...", tokens_per_chunk=256, tokens_overlap=32)

# 4) Markdown header split
# from textSplitter import split_markdown
# md = "# Title\n\n## Section A\ncontent\n\n## Section B\ncontent"
# docs = split_markdown(md)

# 5) Code-aware split
# from textSplitter import split_code, Language
# docs = split_code(open("app.py").read(), language=Language.PYTHON, chunk_size=400, chunk_overlap=40)

# 6) JSON recursive split (from URL)
# from textSplitter import split_json_from_url
# docs = split_json_from_url("https://api.smith.langchain.com/openapi.json", max_chunk_size=300)

# 7) HTML header split (from URL)
# from textSplitter import split_html_from_url
# docs = split_html_from_url("https://www.drsudip.com")

# 8) Auto router
# from textSplitter import split_auto, SplitConfig, Language

# # PDF path
# pdf_docs = split_auto("attention.pdf", SplitConfig(600, 60))

# # Raw text (treat as code)
# code_docs = split_auto(open("app.py").read(), SplitConfig(400, 40), assume_language=Language.PYTHON)