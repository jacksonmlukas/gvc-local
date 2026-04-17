"""Convert solver traces into supervised fine-tuning datasets.

Reads JSONL trace files produced by the GVC / Snap-GVC evaluation harness and
converts them into chat-format datasets suitable for SFTTrainer.  Each line in
the input JSONL is one complete puzzle solve containing agent messages, board
state at each turn, and the final outcome.

Expected trace schema (per line)::

    {
        "puzzle_id": int,
        "difficulty": "easy" | "medium" | "hard" | "very_hard",
        "board": ["WORD1", ..., "WORD16"],
        "groups": [
            {"category": "LABEL", "level": int, "members": ["W1", "W2", "W3", "W4"]}
        ],
        "turns": [
            {
                "agent": "guesser" | "validator" | "snap_guesser" | "consensus",
                "role": "system" | "user" | "assistant",
                "content": "...",
                "remaining_words": ["..."],
                "guess": ["W1", "W2", "W3", "W4"] | null,
                "category_label": "LABEL" | null,
                "correct": bool | null
            }
        ],
        "solved": bool,
        "wrong_guesses": int
    }

Usage::

    python -m gvc_local.finetune.data_prep \\
        --trace-path data/traces/solver_traces.jsonl \\
        --output-dir data/sft \\
        --roles guesser validator snap_guesser
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

AgentRole = Literal["guesser", "validator", "snap_guesser", "consensus"]


@dataclass
class Turn:
    """A single agent turn inside a solver trace."""

    agent: AgentRole
    role: Literal["system", "user", "assistant"]
    content: str
    remaining_words: list[str] = field(default_factory=list)
    guess: list[str] | None = None
    category_label: str | None = None
    correct: bool | None = None


@dataclass
class GroupInfo:
    """One of the four ground-truth groups in a puzzle."""

    category: str
    level: int
    members: list[str]


@dataclass
class TraceRecord:
    """Type-safe representation of a single puzzle-solve trace."""

    puzzle_id: int
    difficulty: str
    board: list[str]
    groups: list[GroupInfo]
    turns: list[Turn]
    solved: bool
    wrong_guesses: int

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict) -> TraceRecord:
        """Construct from a parsed JSON dict, validating required keys."""
        return cls(
            puzzle_id=raw["puzzle_id"],
            difficulty=raw.get("difficulty", "unknown"),
            board=raw["board"],
            groups=[GroupInfo(**g) for g in raw["groups"]],
            turns=[Turn(**t) for t in raw["turns"]],
            solved=raw["solved"],
            wrong_guesses=raw.get("wrong_guesses", 0),
        )


# ---------------------------------------------------------------------------
# System prompts (mirroring the upstream GVC / Snap-GVC agents)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "guesser": (
        "You are an Expert Word Grouping Agent for the NYT Connections puzzle. "
        "You deeply understand literature, culture, and wordplay. Given 16 words on a "
        "board, propose a group of exactly 4 related words and a specific category label "
        "that another agent could use to independently re-derive the same group."
    ),
    "validator": (
        "You are an Expert Word Grouping Validator. Given a word bank and a category "
        "label, identify exactly 4 words from the bank that best fit the category. "
        "You must work independently from the Guesser -- derive the group solely from "
        "the label and the remaining words."
    ),
    "snap_guesser": (
        "You are a Snap Guesser for the NYT Connections puzzle. You break analysis "
        "paralysis by making quick, intuitive groupings. Given remaining words, propose "
        "one group of exactly 4 words with a short reason. Trust your gut."
    ),
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _build_conversations_for_role(
    record: TraceRecord,
    role: AgentRole,
) -> list[list[dict[str, str]]]:
    """Extract chat conversations for *role* from a single trace.

    Returns a list of conversations (one per consensus-reaching exchange) where
    each conversation is a list of ``{"role": ..., "content": ...}`` dicts.
    """
    system_msg = SYSTEM_PROMPTS.get(role)
    if system_msg is None:
        return []

    conversations: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
    in_scope = False

    for turn in record.turns:
        if turn.agent != role:
            # When we switch away from the target role after accumulating turns,
            # close the conversation if it ended with an assistant reply.
            if in_scope and len(current) >= 3 and current[-1]["role"] == "assistant":
                conversations.append(current)
                current = [{"role": "system", "content": system_msg}]
                in_scope = False
            continue

        in_scope = True
        current.append({"role": turn.role, "content": turn.content})

    # Flush the last conversation if it ends with an assistant message.
    if in_scope and len(current) >= 3 and current[-1]["role"] == "assistant":
        conversations.append(current)

    return conversations


def _passes_filter(
    record: TraceRecord,
    include_near_misses: bool = False,
) -> bool:
    """Return True if the trace should be included in the SFT dataset."""
    if record.solved:
        return True
    if include_near_misses and record.wrong_guesses <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_traces(trace_path: str | Path) -> list[TraceRecord]:
    """Load and parse a JSONL trace file into ``TraceRecord`` objects."""
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    records: list[TraceRecord] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(TraceRecord.from_dict(raw))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping line %d in %s: %s", lineno, path, exc)
    logger.info("Loaded %d traces from %s", len(records), path)
    return records


def traces_to_conversations(
    records: list[TraceRecord],
    roles: list[AgentRole],
    include_near_misses: bool = False,
) -> list[dict]:
    """Convert filtered traces into ShareGPT-style conversation dicts.

    Returns a list of dicts, each with keys ``"conversations"`` (the chat
    turns) and ``"metadata"`` (puzzle_id, difficulty, agent role).
    """
    samples: list[dict] = []

    for record in records:
        if not _passes_filter(record, include_near_misses=include_near_misses):
            continue

        for role in roles:
            convos = _build_conversations_for_role(record, role)
            for convo in convos:
                samples.append(
                    {
                        "conversations": convo,
                        "metadata": {
                            "puzzle_id": record.puzzle_id,
                            "difficulty": record.difficulty,
                            "agent_role": role,
                        },
                    }
                )

    logger.info(
        "Built %d SFT samples from %d traces (roles=%s, near_misses=%s)",
        len(samples),
        len(records),
        roles,
        include_near_misses,
    )
    return samples


def split_dataset(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split keyed on puzzle difficulty.

    Falls back to a simple random split when a stratum has fewer than 2
    samples (sklearn requirement).
    """
    if not samples:
        return [], []

    difficulties = [s["metadata"]["difficulty"] for s in samples]
    unique_counts = {d: difficulties.count(d) for d in set(difficulties)}
    can_stratify = all(c >= 2 for c in unique_counts.values()) and len(unique_counts) > 1

    if can_stratify:
        train, val = train_test_split(
            samples,
            test_size=val_ratio,
            random_state=seed,
            stratify=difficulties,
        )
    else:
        logger.warning(
            "Cannot stratify (distribution=%s); falling back to random split.",
            unique_counts,
        )
        train, val = train_test_split(samples, test_size=val_ratio, random_state=seed)

    logger.info("Split: %d train / %d val", len(train), len(val))
    return train, val


def save_jsonl(samples: list[dict], path: Path) -> None:
    """Write samples as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


def save_huggingface(
    train: list[dict],
    val: list[dict],
    output_dir: Path,
) -> None:
    """Save as a HuggingFace DatasetDict (arrow format).

    Falls back to JSONL if ``datasets`` is not installed.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        logger.warning("'datasets' not installed; saving as JSONL instead.")
        save_jsonl(train, output_dir / "train.jsonl")
        save_jsonl(val, output_dir / "val.jsonl")
        return

    def _flatten(samples: list[dict]) -> dict[str, list]:
        """Convert list-of-dicts into dict-of-lists for Dataset.from_dict."""
        flat: dict[str, list] = {
            "conversations": [],
            "puzzle_id": [],
            "difficulty": [],
            "agent_role": [],
        }
        for s in samples:
            flat["conversations"].append(json.dumps(s["conversations"], ensure_ascii=False))
            flat["puzzle_id"].append(s["metadata"]["puzzle_id"])
            flat["difficulty"].append(s["metadata"]["difficulty"])
            flat["agent_role"].append(s["metadata"]["agent_role"])
        return flat

    ds = DatasetDict(
        {
            "train": Dataset.from_dict(_flatten(train)),
            "validation": Dataset.from_dict(_flatten(val)),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir / "hf_dataset"))
    logger.info("Saved HuggingFace DatasetDict to %s/hf_dataset", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("data-prep")
@click.option(
    "--trace-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the solver traces JSONL file.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/sft",
    show_default=True,
    help="Directory where processed datasets are written.",
)
@click.option(
    "--roles",
    multiple=True,
    default=["guesser", "validator", "snap_guesser"],
    show_default=True,
    help="Agent roles to extract conversations for.",
)
@click.option("--val-ratio", type=float, default=0.1, show_default=True)
@click.option("--include-near-misses", is_flag=True, default=False, show_default=True)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "hf", "both"]),
    default="both",
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
def main(
    trace_path: str,
    output_dir: str,
    roles: tuple[str, ...],
    val_ratio: float,
    include_near_misses: bool,
    fmt: str,
    seed: int,
) -> None:
    """Convert solver traces into SFT chat datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    records = load_traces(trace_path)
    samples = traces_to_conversations(
        records,
        roles=list(roles),  # type: ignore[arg-type]
        include_near_misses=include_near_misses,
    )
    train, val = split_dataset(samples, val_ratio=val_ratio, seed=seed)

    out = Path(output_dir)

    if fmt in ("jsonl", "both"):
        save_jsonl(train, out / "train.jsonl")
        save_jsonl(val, out / "val.jsonl")

    if fmt in ("hf", "both"):
        save_huggingface(train, val, out)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
