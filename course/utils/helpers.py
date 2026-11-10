"""
RAG From Scratch - Display and Helper Utilities
Provides beginner-friendly formatting, inspection, and dataset loading tools.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np


def get_data_dir() -> Path:
    """Return the absolute path to the course/data directory."""
    # Assuming this file is at course/utils/helpers.py
    return Path(__file__).resolve().parent.parent / "data"


def load_eval_questions() -> list[dict[str, Any]]:
    """Load the 14 reference evaluation questions from course/data/eval_questions.json."""
    path = get_data_dir() / "eval_questions.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_docs(docs: Sequence[Any], max_chars: int = 350) -> None:
    """
    Format and print a list of retrieved Document objects with clear visual dividers.

    Args:
        docs: A list of Document objects (or objects with page_content and metadata).
        max_chars: Maximum characters of page_content to preview per chunk.
    """
    if not docs:
        print("\n[!] No documents retrieved.")
        return

    print(f"\n{'='*70}")
    print(f"  RETRIEVED {len(docs)} DOCUMENT CHUNK{'S' if len(docs) != 1 else ''}")
    print(f"{'='*70}")

    for idx, doc in enumerate(docs, start=1):
        content = getattr(doc, "page_content", str(doc)).strip()
        metadata = getattr(doc, "metadata", {})

        source = metadata.get("source", "Unknown Source")
        # If source is a full path, shorten it to filename for display
        if isinstance(source, str) and ("/" in source or "\\" in source):
            source_display = Path(source).name
        else:
            source_display = str(source)

        preview = content if len(content) <= max_chars else content[:max_chars] + " ... [TRUNCATED]"

        print(f"\n--- [Chunk #{idx}] Source: {source_display} ---")
        if metadata:
            meta_filtered = {k: v for k, v in metadata.items() if k != "source"}
            if meta_filtered:
                print(f"    Metadata: {meta_filtered}")
        print(f"Content:\n{preview}\n")
    print(f"{'='*70}\n")


def print_qa_result(question: str, answer: str, sources: Sequence[Any] | None = None) -> None:
    """Print question, answer, and cited sources with clear formatting."""
    print(f"\n{'#'*70}")
    print(f"QUESTION: {question}")
    print(f"{'-'*70}")
    print(f"ANSWER:\n{answer.strip()}")
    if sources:
        print(f"{'-'*70}")
        print("SOURCES CITED:")
        seen = set()
        for s in sources:
            source_name = getattr(s, "metadata", {}).get("source", str(s))
            if isinstance(source_name, str) and ("/" in source_name or "\\" in source_name):
                source_name = Path(source_name).name
            if source_name not in seen:
                print(f"  * {source_name}")
                seen.add(source_name)
    print(f"{'#'*70}\n")


def cosine_sim(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Compute cosine similarity between two numeric vectors using pure NumPy."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
