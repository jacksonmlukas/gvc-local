"""Tests for the puzzle category tagger.

Covers the keyword rules, structural blank-pattern detection, voting logic,
and the deterministic-tie-break behaviour that keeps cross-run stratum labels
stable.
"""

from __future__ import annotations

from gvc_local.eval.tagger import (
    DEFAULT_STRATUM,
    _classify_group_text,
    tag_puzzle,
    tag_puzzles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_puzzle(*group_texts: str, puzzle_id: int = 0) -> dict:
    return {
        "puzzle_id": puzzle_id,
        "answers": [
            {"level": i, "group": text, "members": ["A", "B", "C", "D"]}
            for i, text in enumerate(group_texts)
        ],
    }


# ---------------------------------------------------------------------------
# _classify_group_text — direct unit tests on the keyword rules
# ---------------------------------------------------------------------------


class TestClassifyGroupText:
    def test_wordplay_anagram(self):
        assert _classify_group_text("Anagrams of fruits") == "wordplay"

    def test_wordplay_homophone(self):
        assert _classify_group_text("Homophones of body parts") == "wordplay"

    def test_silent_letter(self):
        assert _classify_group_text("Words with a silent K") == "silent-letter"

    def test_silent_letter_remove(self):
        assert _classify_group_text("Add a letter to get a tree") == "silent-letter"

    def test_tag_fillin_blanks(self):
        assert _classify_group_text("___ Park") == "tag-fillin"

    def test_tag_fillin_blanks_suffix(self):
        assert _classify_group_text("Star ___") == "tag-fillin"

    def test_cultural_movie(self):
        assert _classify_group_text("Tom Cruise MOVIES") == "cultural"

    def test_cultural_athlete(self):
        assert _classify_group_text("Hall of Fame quarterbacks") == "cultural"

    def test_unknown_returns_none(self):
        assert _classify_group_text("Types of bread") is None

    def test_empty_input(self):
        assert _classify_group_text("") is None
        assert _classify_group_text("   ") is None


# ---------------------------------------------------------------------------
# tag_puzzle — voting + tie-breaking
# ---------------------------------------------------------------------------


class TestTagPuzzle:
    def test_unanimous_wordplay(self):
        puzzle = _mk_puzzle(
            "Anagrams of fish",
            "Homophones of colours",
            "Palindromes",
            "Rhymes with cat",
        )
        assert tag_puzzle(puzzle) == "wordplay"

    def test_majority_wins(self):
        puzzle = _mk_puzzle(
            "Anagrams of fish",  # wordplay
            "Homophones of colours",  # wordplay
            "Tom Hanks movies",  # cultural
            "Types of bread",  # uncertain
        )
        assert tag_puzzle(puzzle) == "wordplay"

    def test_tie_breaks_by_priority(self):
        # 1 vote each for wordplay + silent-letter; priority order picks wordplay.
        puzzle = _mk_puzzle(
            "Anagrams of fish",
            "Silent K words",
            "Types of bread",
            "Kinds of trees",
        )
        # STRATA = ("wordplay", "silent-letter", "tag-fillin", "cultural", "category")
        # wordplay comes before silent-letter, so wordplay wins.
        assert tag_puzzle(puzzle) == "wordplay"

    def test_all_uncertain_falls_back_to_default(self):
        puzzle = _mk_puzzle(
            "Types of bread",
            "Kinds of trees",
            "Things in a kitchen",
            "Tools",
        )
        assert tag_puzzle(puzzle) == DEFAULT_STRATUM

    def test_empty_answers(self):
        assert tag_puzzle({"answers": []}) == DEFAULT_STRATUM

    def test_missing_answers_field(self):
        assert tag_puzzle({}) == DEFAULT_STRATUM

    def test_supports_groups_alias(self):
        # Some upstream formats use "groups" instead of "answers".
        puzzle = {
            "puzzle_id": 1,
            "groups": [{"group": "Anagrams", "members": ["A", "B", "C", "D"]}],
        }
        assert tag_puzzle(puzzle) == "wordplay"


# ---------------------------------------------------------------------------
# tag_puzzles — batch tagging
# ---------------------------------------------------------------------------


class TestTagPuzzles:
    def test_does_not_mutate_input(self):
        original = _mk_puzzle("Anagrams of fish")
        original_copy = dict(original)
        tag_puzzles([original])
        assert original == original_copy

    def test_writes_category_field(self):
        puzzles = [
            _mk_puzzle("Anagrams of fish", puzzle_id=0),
            _mk_puzzle("Tom Hanks movies", puzzle_id=1),
            _mk_puzzle("Types of bread", puzzle_id=2),
        ]
        tagged = tag_puzzles(puzzles)
        assert tagged[0]["category"] == "wordplay"
        assert tagged[1]["category"] == "cultural"
        assert tagged[2]["category"] == DEFAULT_STRATUM
        # Original puzzle_ids preserved.
        assert [t["puzzle_id"] for t in tagged] == [0, 1, 2]

    def test_empty_input(self):
        assert tag_puzzles([]) == []
