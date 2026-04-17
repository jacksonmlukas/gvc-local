"""RAG (Retrieval-Augmented Generation) layer for puzzle-solving agents.

Provides contextual grounding by retrieving similar historical puzzles and
successful solve traces, replacing brute-force context stuffing with
embedding-based retrieval over a FAISS index.

Requires ``faiss-cpu`` and ``sentence-transformers`` to be installed.
The rest of gvc-local works fine without them — RAG features are optional.

Typical usage::

    from gvc_local.rag import PuzzleRetriever

    retriever = PuzzleRetriever.from_index("data/rag_index")
    result = retriever.retrieve(["CRICKET", "FROG", "HARE", "KANGAROO", ...], k=3)
    print(result.format_for_prompt())
"""


def __getattr__(name: str):  # noqa: N807
    """Lazy imports so the package doesn't fail when faiss is missing."""
    if name in ("IndexConfig", "build_index"):
        from gvc_local.rag.indexer import IndexConfig, build_index

        return {"IndexConfig": IndexConfig, "build_index": build_index}[name]
    if name in ("PuzzleRetriever", "RetrievalResult"):
        from gvc_local.rag.retriever import PuzzleRetriever, RetrievalResult

        return {"PuzzleRetriever": PuzzleRetriever, "RetrievalResult": RetrievalResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_index",
    "IndexConfig",
    "PuzzleRetriever",
    "RetrievalResult",
]
