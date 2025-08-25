# utils/pretty_print.py
from typing import Iterable

def print_docs_pretty(docs: Iterable, show_source: bool = True) -> None:
    """
    Nicely print LangChain Document objects line by line.

    Args:
        docs: Iterable of Documents (each having .page_content and .metadata).
        show_source: If True, prints the source header before content.
    """
    if not docs:
        print("[no documents]")
        return

    for idx, doc in enumerate(docs, start=1):
        # Defensive: handle plain strings or dicts that look like Documents
        page_content = getattr(doc, "page_content", None) or str(doc)
        metadata = getattr(doc, "metadata", {}) or {}

        if show_source:
            src = metadata.get("source", f"doc_{idx}")
            print(f"--- Source: {src} ---")
        for line in str(page_content).splitlines():
            print(line)
