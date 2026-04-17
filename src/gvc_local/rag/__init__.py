"""RAG (Retrieval-Augmented Generation) layer for puzzle-solving agents.

Provides contextual grounding by retrieving similar historical puzzles and
successful solve traces, replacing brute-force context stuffing with
embedding-based retrieval over a FAISS index.

Typical usage:

    from gvc_local.rag import PuzzleRetriever

    retriever = PuzzleRetriever.from_index("data/rag_index")
    result = retriever.retrieve(["CRICKET", "FROG", "HARE", "KANGAROO", ...], k=3)
    print(result.format_for_prompt())
"""

from gvc_local.rag.indexer import build_index, IndexConfig
from gvc_local.rag.retriever import PuzzleRetriever, RetrievalResult

__all__ = [
    "build_index",
    "IndexConfig",
    "PuzzleRetriever",
    "RetrievalResult",
]
