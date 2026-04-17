"""Index historical puzzle metadata and solve traces into FAISS for retrieval.

Two document types are indexed into separate namespaces within a single FAISS
index:

1. **Puzzle metadata** -- the 16 words, 4 category labels, and difficulty
   levels from each historical puzzle. Embedded as a single text block so
   word-pattern similarity surfaces naturally.

2. **Solve traces** -- successful agent reasoning chains (what worked for
   similar word patterns). Embedded as-is so retrieval finds traces whose
   language/strategy matches the current board state.

Embeddings are produced by ``sentence-transformers/all-MiniLM-L6-v2`` -- the
same model the original paper (Pandian et al., ACL 2025) uses for cosine
similarity metrics.  The FAISS index uses IVF (inverted-file) partitioning
for efficient approximate search once the corpus grows past a few hundred
puzzles.

Entry point
-----------
    build_index(puzzles_path, traces_path, output_dir)

Or from the CLI:
    python -m scripts.build_rag_index --puzzles data/puzzles.json ...
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# IVF partitions -- rule of thumb is sqrt(n).  We cap the minimum at 4
# (FAISS requires nlist >= 1) and let the builder pick automatically.
_MIN_IVF_NLIST = 4

# Filenames written inside the output directory
_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.pkl"
_CONFIG_FILE = "index_config.json"

# Document type tags stored in metadata so the retriever can filter by type.
DOC_TYPE_PUZZLE = "puzzle"
DOC_TYPE_TRACE = "trace"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IndexConfig:
    """Serialisable configuration for index construction.

    Kept separate from runtime so we can store it alongside the index and
    reproduce the build later.
    """

    embed_model: str = EMBED_MODEL
    embed_dim: int = EMBED_DIM
    nprobe: int = 8  # how many IVF cells to visit at query time
    use_ivf: bool = True
    batch_size: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "nprobe": self.nprobe,
            "use_ivf": self.use_ivf,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndexConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------


@dataclass
class DocRecord:
    """A single document to be indexed, with its embedding text and metadata."""

    text: str
    doc_type: str  # DOC_TYPE_PUZZLE or DOC_TYPE_TRACE
    meta: dict[str, Any] = field(default_factory=dict)


def _puzzle_to_doc(puzzle: dict[str, Any], puzzle_id: int) -> DocRecord:
    """Convert a single puzzle dict (upstream JSON schema) into an indexable doc.

    Expected puzzle schema (matches the Eyefyre/NYT-Connections-Answers repo
    and the upstream ``rsallms.game.Category`` dataclass):

        {
          "answers": [
            {"level": 0, "group": "JUMPING ANIMALS",
             "members": ["CRICKET", "FROG", "HARE", "KANGAROO"]},
            ...
          ]
        }

    The text representation concatenates all words and category labels into a
    single string so the embedding captures the full board state.
    """
    categories = puzzle.get("answers", puzzle.get("groups", []))
    all_words: list[str] = []
    cat_labels: list[str] = []
    levels: list[int] = []

    for cat in categories:
        members = cat.get("members", [])
        all_words.extend(members)
        cat_labels.append(cat.get("group", ""))
        levels.append(cat.get("level", -1))

    # Build a human-readable text block for embedding
    word_str = ", ".join(all_words)
    cat_str = " | ".join(
        f"{cat.get('group', '?')} (level {cat.get('level', '?')}): "
        f"{', '.join(cat.get('members', []))}"
        for cat in categories
    )
    text = f"Puzzle words: {word_str}\nCategories: {cat_str}"

    return DocRecord(
        text=text,
        doc_type=DOC_TYPE_PUZZLE,
        meta={
            "puzzle_id": puzzle_id,
            "words": all_words,
            "categories": cat_labels,
            "levels": levels,
        },
    )


def _trace_to_doc(trace: dict[str, Any]) -> DocRecord:
    """Convert a solve-trace record into an indexable doc.

    Expected trace schema (flexible -- we extract what's available):

        {
          "puzzle_id": 42,
          "solver": "snap_gvc",
          "success": true,
          "reasoning": "Step 1: I noticed CRICKET, FROG, HARE, KANGAROO ...",
          "guesses": [["CRICKET", "FROG", "HARE", "KANGAROO"], ...],
          "words": ["CRICKET", "FROG", ...]
        }
    """
    reasoning = trace.get("reasoning", "")
    guesses = trace.get("guesses", [])
    solver = trace.get("solver", "unknown")
    puzzle_id = trace.get("puzzle_id", -1)
    words = trace.get("words", [])

    # Build the text representation
    parts: list[str] = []
    if words:
        parts.append(f"Board: {', '.join(words)}")
    parts.append(f"Solver: {solver}")
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    if guesses:
        guess_strs = [f"  [{', '.join(g)}]" for g in guesses]
        parts.append("Guesses:\n" + "\n".join(guess_strs))

    text = "\n".join(parts)

    return DocRecord(
        text=text,
        doc_type=DOC_TYPE_TRACE,
        meta={
            "puzzle_id": puzzle_id,
            "solver": solver,
            "words": words,
            "success": trace.get("success", True),
        },
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode a list of strings into L2-normalised float32 vectors."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-norm for cosine via inner-product
    )
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _make_faiss_index(dim: int, n_vectors: int, use_ivf: bool) -> faiss.Index:
    """Create a FAISS index appropriate for the corpus size.

    For small corpora (< 256 vectors) or when ``use_ivf`` is False, a flat
    inner-product index is used.  Otherwise an IVF index with a flat quantiser
    partitions the space for faster search.
    """
    if not use_ivf or n_vectors < 256:
        logger.info("Using flat inner-product index (n=%d)", n_vectors)
        return faiss.IndexFlatIP(dim)

    nlist = max(_MIN_IVF_NLIST, int(np.sqrt(n_vectors)))
    logger.info("Using IVF index: nlist=%d for n=%d vectors", nlist, n_vectors)
    quantiser = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    return index


def build_index(
    puzzles_path: str | Path,
    output_dir: str | Path,
    traces_path: str | Path | None = None,
    config: IndexConfig | None = None,
) -> Path:
    """Build a FAISS index from puzzle data and (optionally) solve traces.

    Parameters
    ----------
    puzzles_path:
        Path to a JSON file containing an array of puzzle objects.  Each object
        must have an ``"answers"`` key with the standard Connections schema
        (see ``_puzzle_to_doc``).
    output_dir:
        Directory where the index, metadata, and config will be written.
        Created if it does not exist.
    traces_path:
        Optional path to a JSON/JSONL file of solve-trace records.  If the
        file extension is ``.jsonl``, it is read line-by-line; otherwise it is
        loaded as a JSON array.
    config:
        Override the default ``IndexConfig``.

    Returns
    -------
    Path to ``output_dir`` (for chaining).
    """
    cfg = config or IndexConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & prepare documents
    # ------------------------------------------------------------------
    logger.info("Loading puzzles from %s", puzzles_path)
    with open(puzzles_path) as f:
        raw_puzzles: list[dict[str, Any]] = json.load(f)
    docs = [_puzzle_to_doc(p, idx) for idx, p in enumerate(raw_puzzles)]
    logger.info("Prepared %d puzzle documents", len(docs))

    if traces_path is not None:
        traces_path = Path(traces_path)
        logger.info("Loading traces from %s", traces_path)
        traces = _load_traces(traces_path)
        trace_docs = [_trace_to_doc(t) for t in traces if t.get("success", True)]
        logger.info("Prepared %d trace documents (successful only)", len(trace_docs))
        docs.extend(trace_docs)

    if not docs:
        raise ValueError("No documents to index -- check your input files.")

    # ------------------------------------------------------------------
    # 2. Embed
    # ------------------------------------------------------------------
    logger.info("Loading embedding model: %s", cfg.embed_model)
    model = SentenceTransformer(cfg.embed_model)
    texts = [d.text for d in docs]
    logger.info("Encoding %d documents ...", len(texts))
    vectors = _embed_texts(texts, model, batch_size=cfg.batch_size)
    assert vectors.shape == (len(docs), cfg.embed_dim), (
        f"Shape mismatch: got {vectors.shape}, expected ({len(docs)}, {cfg.embed_dim})"
    )

    # ------------------------------------------------------------------
    # 3. Build FAISS index
    # ------------------------------------------------------------------
    index = _make_faiss_index(cfg.embed_dim, len(docs), cfg.use_ivf)

    if hasattr(index, "train") and not index.is_trained:
        logger.info("Training IVF index ...")
        index.train(vectors)

    index.add(vectors)
    logger.info("Index contains %d vectors", index.ntotal)

    # ------------------------------------------------------------------
    # 4. Persist
    # ------------------------------------------------------------------
    faiss.write_index(index, str(output / _INDEX_FILE))

    metadata = [{"doc_type": d.doc_type, **d.meta} for d in docs]
    with open(output / _META_FILE, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output / _CONFIG_FILE, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logger.info("Index written to %s", output)
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_traces(path: Path) -> list[dict[str, Any]]:
    """Load traces from JSON array or JSONL (one JSON object per line)."""
    if path.suffix == ".jsonl":
        traces: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
