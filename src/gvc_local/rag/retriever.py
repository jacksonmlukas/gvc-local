"""Retrieve relevant historical puzzles and solve traces for agent grounding.

Given a set of remaining puzzle words, the retriever finds:

1. **Similar puzzles** -- historical boards with overlapping word patterns or
   thematic similarity, ranked by embedding cosine similarity.
2. **Relevant traces** -- successful reasoning chains from agents that solved
   similar boards, so the current agent can learn from prior strategies.

The retrieved context is formatted as structured text ready to inject into
agent system/user prompts.

Usage
-----
    retriever = PuzzleRetriever.from_index("data/rag_index")
    result = retriever.retrieve(["CRICKET", "FROG", "HARE", ...], k=3)
    prompt_ctx = result.format_for_prompt()
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from gvc_local.rag.indexer import (
    _CONFIG_FILE,
    _INDEX_FILE,
    _META_FILE,
    DOC_TYPE_PUZZLE,
    DOC_TYPE_TRACE,
    IndexConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoredPuzzle:
    """A single retrieved puzzle with its similarity score."""

    puzzle_id: int
    words: list[str]
    categories: list[str]
    levels: list[int]
    score: float

    def format(self) -> str:
        """Human-readable single-puzzle summary for prompt injection."""
        cat_lines = []
        for cat, lvl in zip(self.categories, self.levels, strict=False):
            cat_lines.append(f"  - {cat} (difficulty {lvl})")
        cats = "\n".join(cat_lines)
        words = ", ".join(self.words)
        return (
            f"Puzzle #{self.puzzle_id} (similarity {self.score:.3f}):\n"
            f"  Words: {words}\n"
            f"  Categories:\n{cats}"
        )


@dataclass
class ScoredTrace:
    """A single retrieved solve trace with its similarity score."""

    puzzle_id: int
    solver: str
    words: list[str]
    success: bool
    score: float
    # The full metadata dict for extensibility
    raw: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        """Human-readable trace summary for prompt injection."""
        words = ", ".join(self.words) if self.words else "(unknown)"
        return (
            f"Trace from puzzle #{self.puzzle_id} "
            f"[{self.solver}, {'solved' if self.success else 'failed'}] "
            f"(similarity {self.score:.3f}):\n"
            f"  Board: {words}"
        )


@dataclass
class RetrievalResult:
    """Container for a retrieval query's output."""

    query_words: list[str]
    similar_puzzles: list[ScoredPuzzle] = field(default_factory=list)
    relevant_traces: list[ScoredTrace] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format the full retrieval result as context text for agent prompts.

        Returns a structured block that can be directly inserted into a system
        message or user prompt to ground the agent's reasoning.
        """
        sections: list[str] = []

        if self.similar_puzzles:
            puzzle_block = "## Similar Historical Puzzles\n\n" + "\n\n".join(
                p.format() for p in self.similar_puzzles
            )
            sections.append(puzzle_block)

        if self.relevant_traces:
            trace_block = "## Relevant Solve Traces\n\n" + "\n\n".join(
                t.format() for t in self.relevant_traces
            )
            sections.append(trace_block)

        if not sections:
            return "(No relevant historical context found.)"

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (useful for logging / debugging)."""
        return {
            "query_words": self.query_words,
            "similar_puzzles": [
                {
                    "puzzle_id": p.puzzle_id,
                    "words": p.words,
                    "categories": p.categories,
                    "score": p.score,
                }
                for p in self.similar_puzzles
            ],
            "relevant_traces": [
                {
                    "puzzle_id": t.puzzle_id,
                    "solver": t.solver,
                    "score": t.score,
                }
                for t in self.relevant_traces
            ],
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class PuzzleRetriever:
    """Load a pre-built FAISS index and retrieve context for puzzle agents.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index (already loaded / trained).
    metadata : list[dict]
        Per-vector metadata aligned by position with the index.
    model : SentenceTransformer
        Encoder used at query time (must match the one used at index time).
    config : IndexConfig
        The config that was used to build the index.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict[str, Any]],
        model: SentenceTransformer,
        config: IndexConfig,
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.model = model
        self.config = config

        # Pre-compute document-type masks for filtered retrieval
        self._puzzle_ids = np.array(
            [i for i, m in enumerate(metadata) if m.get("doc_type") == DOC_TYPE_PUZZLE],
            dtype=np.int64,
        )
        self._trace_ids = np.array(
            [i for i, m in enumerate(metadata) if m.get("doc_type") == DOC_TYPE_TRACE],
            dtype=np.int64,
        )
        logger.info(
            "PuzzleRetriever ready: %d puzzles, %d traces in index",
            len(self._puzzle_ids),
            len(self._trace_ids),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_index(cls, index_dir: str | Path) -> PuzzleRetriever:
        """Load a retriever from a directory written by ``build_index``.

        Parameters
        ----------
        index_dir:
            Path to the directory containing ``faiss.index``,
            ``metadata.pkl``, and ``index_config.json``.
        """
        index_dir = Path(index_dir)

        with open(index_dir / _CONFIG_FILE) as f:
            config = IndexConfig.from_dict(json.load(f))

        logger.info("Loading FAISS index from %s", index_dir / _INDEX_FILE)
        index = faiss.read_index(str(index_dir / _INDEX_FILE))

        # Set nprobe for IVF indices
        if hasattr(index, "nprobe"):
            index.nprobe = config.nprobe

        with open(index_dir / _META_FILE, "rb") as f:
            metadata: list[dict[str, Any]] = pickle.load(f)

        logger.info("Loading embedding model: %s", config.embed_model)
        model = SentenceTransformer(config.embed_model)

        return cls(index=index, metadata=metadata, model=model, config=config)

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        words: list[str],
        k: int = 5,
        k_puzzles: int | None = None,
        k_traces: int | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant context for a set of remaining puzzle words.

        Parameters
        ----------
        words:
            The remaining words on the board (1--16 strings).
        k:
            Default number of results per document type.  Overridden by
            ``k_puzzles`` / ``k_traces`` when provided.
        k_puzzles:
            Number of similar puzzles to retrieve.
        k_traces:
            Number of relevant solve traces to retrieve.

        Returns
        -------
        A ``RetrievalResult`` containing scored puzzles and traces.
        """
        k_p = k_puzzles if k_puzzles is not None else k
        k_t = k_traces if k_traces is not None else k

        query_text = f"Puzzle words: {', '.join(words)}"
        query_vec = self._encode(query_text)

        similar_puzzles = self._search_subset(query_vec, self._puzzle_ids, k_p, DOC_TYPE_PUZZLE)
        relevant_traces = self._search_subset(query_vec, self._trace_ids, k_t, DOC_TYPE_TRACE)

        return RetrievalResult(
            query_words=words,
            similar_puzzles=[self._meta_to_puzzle(idx, score) for idx, score in similar_puzzles],
            relevant_traces=[self._meta_to_trace(idx, score) for idx, score in relevant_traces],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single query string into a normalised vector."""
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32)

    def _search_subset(
        self,
        query_vec: np.ndarray,
        subset_ids: np.ndarray,
        k: int,
        doc_type: str,
    ) -> list[tuple[int, float]]:
        """Search the FAISS index restricted to a subset of vector IDs.

        Uses an ``IDSelectorArray`` so FAISS only scores vectors of the
        requested document type, avoiding post-hoc filtering that wastes the
        top-k budget.

        Falls back to full search + post-filter when the index type does not
        support ID selectors (e.g., some flat indices in older FAISS builds).
        """
        if len(subset_ids) == 0 or k <= 0:
            return []

        k_actual = min(k, len(subset_ids))

        try:
            sel = faiss.IDSelectorArray(subset_ids)
            params = (
                faiss.SearchParametersIVF(sel=sel)
                if hasattr(self.index, "nprobe")
                else faiss.SearchParameters(sel=sel)
            )
            scores, ids = self.index.search(query_vec, k_actual, params=params)
        except (RuntimeError, AttributeError):
            # Fallback: search a wider set and post-filter
            logger.debug("ID-selector search unavailable, falling back to post-filter")
            scores, ids = self._search_with_postfilter(query_vec, subset_ids, k_actual, doc_type)

        results: list[tuple[int, float]] = []
        for score, vid in zip(scores[0], ids[0], strict=False):
            if vid == -1:
                continue
            results.append((int(vid), float(score)))
        return results

    def _search_with_postfilter(
        self,
        query_vec: np.ndarray,
        subset_ids: np.ndarray,
        k: int,
        doc_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force fallback: over-fetch then filter by doc_type."""
        subset_set = set(subset_ids.tolist())
        # Fetch more than k to compensate for filtering
        fetch_k = min(self.index.ntotal, k * 5)
        all_scores, all_ids = self.index.search(query_vec, fetch_k)

        filtered_scores: list[float] = []
        filtered_ids: list[int] = []
        for score, vid in zip(all_scores[0], all_ids[0], strict=False):
            if vid == -1:
                continue
            if int(vid) in subset_set:
                filtered_scores.append(score)
                filtered_ids.append(int(vid))
                if len(filtered_ids) >= k:
                    break

        # Pad to k with -1 / -inf if needed
        while len(filtered_ids) < k:
            filtered_ids.append(-1)
            filtered_scores.append(-float("inf"))

        return (
            np.array([filtered_scores], dtype=np.float32),
            np.array([filtered_ids], dtype=np.int64),
        )

    def _meta_to_puzzle(self, idx: int, score: float) -> ScoredPuzzle:
        m = self.metadata[idx]
        return ScoredPuzzle(
            puzzle_id=m.get("puzzle_id", -1),
            words=m.get("words", []),
            categories=m.get("categories", []),
            levels=m.get("levels", []),
            score=score,
        )

    def _meta_to_trace(self, idx: int, score: float) -> ScoredTrace:
        m = self.metadata[idx]
        return ScoredTrace(
            puzzle_id=m.get("puzzle_id", -1),
            solver=m.get("solver", "unknown"),
            words=m.get("words", []),
            success=m.get("success", True),
            score=score,
            raw=m,
        )
