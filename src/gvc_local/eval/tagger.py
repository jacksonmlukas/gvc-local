"""Heuristic puzzle category tagger for NYT Connections.

The eval harness in :mod:`gvc_local.eval_harness` relies on puzzles being
labelled with a ``strata`` field so that :func:`stratified_sample` can produce
balanced coverage across puzzle types.  Raw puzzles from the upstream
NYT-Connections-Answers dataset are unlabelled.  This module provides a
keyword-based heuristic classifier that maps each puzzle's four group
descriptions into a single stratum label.

Taxonomy (mirrors the categories used in Pandian et al., ACL 2025 REALM)
------------------------------------------------------------------------
1. **wordplay**       — homophones, anagrams, palindromes, rhymes, hidden words.
2. **silent-letter**  — silent letters, "missing" letters, removed-letter clues.
3. **tag-fillin**     — "___ X" or "X ___" formula categories where a tag word
                        completes a phrase.
4. **cultural**       — proper nouns, named entities, branded references that
                        require external cultural knowledge.
5. **category**       — generic semantic groupings (types of, kinds of, etc.).
                        The catch-all bucket.

The classifier is intentionally simple — it is not the contribution of the
project.  The contribution is that strata exist *at all*, so the eval harness
can report per-stratum performance and stratify the sample.  A noisier but
unbiased tagger is fine for our use case: bootstrap CIs on a stratified sample
of a noisy taxonomy still reflect honest uncertainty.

Usage
-----
Programmatic::

    from gvc_local.eval.tagger import tag_puzzle, tag_puzzles
    puzzle = {"answers": [...]}
    label = tag_puzzle(puzzle)
    tagged = tag_puzzles(list_of_puzzles)

CLI (one-shot tagging of an upstream dump)::

    python -m gvc_local.eval.tagger \\
        --input data/puzzles/raw_connections.json \\
        --output data/puzzles/tagged_connections.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

STRATA = ("wordplay", "silent-letter", "tag-fillin", "cultural", "category")
DEFAULT_STRATUM = "category"


# Keywords applied to the upper-cased group description.  Order matters — earlier
# rules win.  Tunable; the classifier is intentionally conservative.
_RULES: list[tuple[str, list[str]]] = [
    (
        "silent-letter",
        [
            "SILENT",
            "MISSING LETTER",
            "REMOVE A LETTER",
            "REMOVE LETTER",
            "DROP A LETTER",
            "DROP LETTER",
            "ADD A LETTER",
            "ADD LETTER",
            "INSERT A LETTER",
            "INSERT LETTER",
            "MINUS A LETTER",
            "MINUS LETTER",
        ],
    ),
    (
        "wordplay",
        [
            "ANAGRAM",
            "HOMOPHONE",
            "PALINDROME",
            "RHYME",
            "RHYMING",
            "PUN",
            "HIDDEN WORD",
            "HIDDEN INSIDE",
            "CONTAINS",
            "SPELLED BACKWARD",
            "SPELLED BACKWARDS",
            "DOUBLE LETTER",
            "DOUBLE LETTERS",
            "DOUBLED LETTER",
            "PORTMANTEAU",
            "SOUND-ALIKE",
            "SOUND ALIKE",
            "SOUNDS LIKE",
        ],
    ),
]

# "Tag fill-in" is detected structurally rather than by keyword: a group
# description containing "___" (blank) or starting/ending with a one-word tag.
_BLANK_PATTERN = re.compile(r"_{2,}")


# Cultural-knowledge cues: capitalised proper nouns inside the group description,
# brand names, etc.  Used only as a secondary signal after rules above.
_CULTURAL_KEYWORDS = (
    "CHARACTER",
    "MOVIE",
    "SONG",
    "ALBUM",
    "BAND",
    "ACTOR",
    "ACTRESS",
    "DIRECTOR",
    "AUTHOR",
    "BOOK",
    "TV SHOW",
    "SHOW TITLE",
    "ATHLETE",
    "QUARTERBACK",
    "PRESIDENT",
    "BRAND",
    "COMPANY",
    "BAR",
    "TEAM",
    "CELEBRITY",
)


# ---------------------------------------------------------------------------
# Single-puzzle tagging
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().upper()


def _classify_group_text(group_text: str) -> str | None:
    """Return a stratum label for a single group description, or None if uncertain."""
    text = _normalise(group_text)
    if not text:
        return None

    # Rule-based keyword sweep
    for label, keywords in _RULES:
        for kw in keywords:
            if kw in text:
                return label

    # Structural tag-fillin detection: "___" anywhere in the group description.
    if _BLANK_PATTERN.search(text):
        return "tag-fillin"

    # Cultural-knowledge cues
    for kw in _CULTURAL_KEYWORDS:
        if kw in text:
            return "cultural"

    return None


def tag_puzzle(puzzle: dict[str, Any]) -> str:
    """Tag a single puzzle with a stratum label.

    The puzzle is expected to have an ``answers`` field — a list of group
    dicts each with at minimum a ``group`` key (the human-readable category
    description).  Compatible with the upstream rsallms format and the
    NYT-Connections-Answers JSON.

    Voting rule: each of the 4 groups is classified individually.  The puzzle's
    final stratum is the most frequent non-None classification; ties are broken
    by the priority order in :data:`STRATA`; full uncertainty falls back to
    ``DEFAULT_STRATUM``.
    """
    groups = puzzle.get("answers") or puzzle.get("groups") or []
    votes: list[str] = []

    for group in groups:
        label = _classify_group_text(group.get("group", ""))
        if label is not None:
            votes.append(label)

    if not votes:
        return DEFAULT_STRATUM

    # Most frequent label wins; ties broken by STRATA priority order.
    counter = Counter(votes)
    top_count = max(counter.values())
    tied = [label for label, n in counter.items() if n == top_count]
    if len(tied) == 1:
        return tied[0]
    for label in STRATA:
        if label in tied:
            return label
    return DEFAULT_STRATUM  # unreachable


def tag_puzzles(puzzles: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag every puzzle in an iterable; return new dicts with a ``category`` field set.

    Does not mutate the input.  The output uses ``"category"`` to match the
    field name expected by :func:`gvc_local.eval.harness.categorise_puzzle`.
    """
    tagged: list[dict[str, Any]] = []
    for puzzle in puzzles:
        label = tag_puzzle(puzzle)
        # Preserve the original puzzle dict, just add/overwrite ``category``.
        new_puzzle = dict(puzzle)
        new_puzzle["category"] = label
        tagged.append(new_puzzle)
    return tagged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(path: Path) -> list[dict[str, Any]]:
    """Read puzzles from JSON (array-of-objects) or JSONL."""
    text = path.read_text()
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_output(puzzles: list[dict[str, Any]], path: Path, jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w") as fh:
            for puzzle in puzzles:
                fh.write(json.dumps(puzzle) + "\n")
    else:
        with path.open("w") as fh:
            json.dump(puzzles, fh, indent=2)


def _stratum_distribution(puzzles: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(p.get("category", DEFAULT_STRATUM) for p in puzzles)
    return dict(counter.most_common())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw puzzles JSON or JSONL.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write tagged puzzles.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL output (default: JSON array).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    puzzles = _read_input(args.input)
    logger.info("Loaded %d puzzles from %s", len(puzzles), args.input)
    tagged = tag_puzzles(puzzles)
    _write_output(tagged, args.output, jsonl=args.jsonl)

    dist = _stratum_distribution(tagged)
    logger.info("Wrote %d tagged puzzles to %s", len(tagged), args.output)
    logger.info("Stratum distribution: %s", dist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
