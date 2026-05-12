#!/usr/bin/env python3
"""Prepare fine-tuning data from Connections puzzles.

Converts raw Connections puzzle data into chat-format JSONL for LoRA
fine-tuning on Together AI.  Each puzzle produces up to 4 training
examples (one per round), teaching the model to:

1. Understand the full board layout.
2. Identify the most confident group first.
3. Handle progressively smaller boards as groups are solved.

The output format matches Together AI / OpenAI chat fine-tuning::

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage::

    # Default: 80/20 train/test split, all puzzles
    python scripts/data_prep.py

    # Limit to first 200 puzzles, custom output dir
    python scripts/data_prep.py --max-puzzles 200 --out-dir data/finetune

    # Single-round mode (1 example per puzzle, first guess only)
    python scripts/data_prep.py --single-round
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Add src to path so we can import gvc_local
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gvc_local.game import Category, load_games  # noqa: E402

# ---------------------------------------------------------------------------
# System prompt (simplified from guesser.py for fine-tuning)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at the NYT Connections puzzle. Given a list of 16 words \
(or fewer if some groups have been solved), identify groups of 4 related \
words and their category names.

You MUST format your response EXACTLY as follows:

<UNDERSTANDING_OF_BOARD>
Group1: word1, word2, word3, word4
Group2: word5, word6, word7, word8
Group3: word9, word10, word11, word12
Group4: word13, word14, word15, word16
<END_UNDERSTANDING_OF_BOARD>

<GUESS_FOR_THIS_ROUND>
Group: word_a, word_b, word_c, word_d
Category: category_name
<END_GUESS_FOR_THIS_ROUND>

Pick the group you are MOST confident about as your guess."""


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------


def _build_user_prompt(words: list[str]) -> str:
    """Build the user prompt from remaining words."""
    shuffled = words.copy()
    random.shuffle(shuffled)
    return f"Words: {', '.join(shuffled)}"


def _build_assistant_response(
    all_groups: list[Category],
    guess_group: Category,
) -> str:
    """Build the ideal assistant response for a given board state and guess.

    Parameters
    ----------
    all_groups:
        All remaining groups on the board (for the UNDERSTANDING section).
    guess_group:
        The specific group the model should guess this round.
    """
    # Understanding section: all remaining groups
    understanding_lines = []
    for i, group in enumerate(all_groups, 1):
        members = ", ".join(group.members)
        understanding_lines.append(f"Group{i}: {members}")

    understanding = "\n".join(understanding_lines)

    # Guess section: the target group
    guess_members = ", ".join(guess_group.members)

    return (
        f"<UNDERSTANDING_OF_BOARD>\n"
        f"{understanding}\n"
        f"<END_UNDERSTANDING_OF_BOARD>\n"
        f"\n"
        f"<GUESS_FOR_THIS_ROUND>\n"
        f"Group: {guess_members}\n"
        f"Category: {guess_group.group}\n"
        f"<END_GUESS_FOR_THIS_ROUND>"
    )


def puzzle_to_examples(
    categories: list[Category],
    *,
    single_round: bool = False,
) -> list[dict]:
    """Convert a single puzzle into one or more chat training examples.

    Parameters
    ----------
    categories:
        The 4 categories for this puzzle.
    single_round:
        If True, only produce 1 example (full board → easiest group).
        If False, produce up to 4 examples simulating multi-round play.

    Returns
    -------
    List of dicts, each with a ``"messages"`` key.
    """
    # Sort by level (0 = easiest → 3 = hardest)
    sorted_cats = sorted(categories, key=lambda c: c.level)
    examples = []

    remaining = list(sorted_cats)  # mutable copy

    rounds = 1 if single_round else len(remaining)
    for _i in range(rounds):
        if not remaining:
            break

        # The guess target is the easiest remaining group
        guess_target = remaining[0]

        # All words still on the board
        all_words = [w for cat in remaining for w in cat.members]

        user_prompt = _build_user_prompt(all_words)
        assistant_response = _build_assistant_response(remaining, guess_target)

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        examples.append(example)

        # Remove the guessed group for the next round
        remaining = remaining[1:]

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Connections puzzles as fine-tuning data."
    )
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=None,
        help="Limit the number of puzzles to use (default: all).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of puzzles reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--single-round",
        action="store_true",
        help="Only generate 1 example per puzzle (first guess).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/finetune",
        help="Output directory for train/test JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all puzzles
    print("Loading puzzles from GitHub data source...")
    games = load_games()
    print(f"Loaded {len(games)} puzzles.")

    if args.max_puzzles is not None:
        games = games[: args.max_puzzles]
        print(f"Limiting to first {args.max_puzzles} puzzles.")

    # Train/test split (by puzzle, not by example)
    n_test = max(1, int(len(games) * args.test_frac))
    n_train = len(games) - n_test

    # Shuffle puzzle indices for random split
    indices = list(range(len(games)))
    random.shuffle(indices)

    train_indices = set(indices[:n_train])

    # Generate examples
    train_examples = []
    test_examples = []

    for idx in range(len(games)):
        game = games[idx]
        cats = game._og_groups
        examples = puzzle_to_examples(cats, single_round=args.single_round)

        if idx in train_indices:
            train_examples.extend(examples)
        else:
            test_examples.extend(examples)

    # Shuffle training examples
    random.shuffle(train_examples)

    # Write output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    def _estimate_tokens(examples: list[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(
            len(m["content"]) for ex in examples for m in ex["messages"]
        )
        return total_chars // 4

    train_tokens = _estimate_tokens(train_examples)
    test_tokens = _estimate_tokens(test_examples)

    print(f"\n{'='*50}")
    print(f"Train: {len(train_examples)} examples ({n_train} puzzles)")
    print(f"Test:  {len(test_examples)} examples ({n_test} puzzles)")
    print(f"Train tokens (est): ~{train_tokens:,}")
    print(f"Test tokens (est):  ~{test_tokens:,}")
    print(f"Estimated training cost @ $0.008/1K tokens: ~${train_tokens * 0.008 / 1000:.2f}")
    print("\nFiles written:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
