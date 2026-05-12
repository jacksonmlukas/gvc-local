"""Tests for agent prompt construction and response parsing."""

from __future__ import annotations

import pytest

from gvc_local.agents.guesser import GuesserAgent
from gvc_local.agents.snap_guesser import SnapGuesserAgent
from gvc_local.agents.validator import ValidatorAgent


# ---------------------------------------------------------------------------
# GuesserAgent tests
# ---------------------------------------------------------------------------


class TestGuesserParsing:
    """Test GuesserAgent._parse_reply with various LLM outputs."""

    def test_standard_reply(self):
        reply = """
<UNDERSTANDING_OF_BOARD>
Group1: APPLE, BANANA, CHERRY, DATE
Group2: RED, BLUE, GREEN, YELLOW
Group3: CAT, DOG, FISH, BIRD
Group4: ONE, TWO, THREE, FOUR
<END_UNDERSTANDING_OF_BOARD>

<GUESS_FOR_THIS_ROUND>
Group: APPLE, BANANA, CHERRY, DATE
Category: FRUITS
<END_GUESS_FOR_THIS_ROUND>
"""
        group, category, understanding = GuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert category == "FRUITS"
        assert len(understanding) == 4
        assert understanding[0] == ["APPLE", "BANANA", "CHERRY", "DATE"]

    def test_extra_whitespace(self):
        reply = """
<UNDERSTANDING_OF_BOARD>
  Group 1:  APPLE ,  BANANA ,  CHERRY ,  DATE
  Group 2: RED, BLUE, GREEN, YELLOW
  Group 3: CAT, DOG, FISH, BIRD
  Group 4: ONE, TWO, THREE, FOUR
<END_UNDERSTANDING_OF_BOARD>

<GUESS_FOR_THIS_ROUND>
Group:  APPLE ,  BANANA ,  CHERRY ,  DATE
Category:  FRUITS
<END_GUESS_FOR_THIS_ROUND>
"""
        group, category, understanding = GuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert category == "FRUITS"

    def test_missing_group_raises(self):
        reply = """
<GUESS_FOR_THIS_ROUND>
Category: FRUITS
<END_GUESS_FOR_THIS_ROUND>
"""
        with pytest.raises(ValueError, match="Group"):
            GuesserAgent._parse_reply(reply)

    def test_missing_category_raises(self):
        reply = """
<GUESS_FOR_THIS_ROUND>
Group: APPLE, BANANA, CHERRY, DATE
<END_GUESS_FOR_THIS_ROUND>
"""
        with pytest.raises(ValueError, match="Category"):
            GuesserAgent._parse_reply(reply)

    def test_wrong_word_count_raises(self):
        reply = """
<GUESS_FOR_THIS_ROUND>
Group: APPLE, BANANA, CHERRY
Category: FRUITS
<END_GUESS_FOR_THIS_ROUND>
"""
        with pytest.raises(ValueError, match="Expected 4"):
            GuesserAgent._parse_reply(reply)

    def test_fallback_without_tags(self):
        reply = """
Group: APPLE, BANANA, CHERRY, DATE
Category: FRUITS
"""
        group, category, understanding = GuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert category == "FRUITS"
        assert understanding == []  # no UNDERSTANDING_OF_BOARD tags


class TestGuesserPrompt:
    """Test GuesserAgent._build_prompt with various inputs."""

    def test_minimal_prompt(self):
        prompt = GuesserAgent._build_prompt(
            None,  # self not used for static-like call
            remaining_words=["APPLE", "BANANA"],
            feedback="",
            rag_context="",
            validator_feedback="",
            previous_understanding=None,
        )
        assert "APPLE, BANANA" in prompt
        assert "Remaining Words" in prompt

    def test_with_feedback(self):
        prompt = GuesserAgent._build_prompt(
            None,
            remaining_words=["APPLE"],
            feedback="Some feedback here",
            rag_context="",
            validator_feedback="",
            previous_understanding=None,
        )
        assert "Game Engine Feedback" in prompt
        assert "Some feedback here" in prompt

    def test_with_previous_understanding(self):
        prompt = GuesserAgent._build_prompt(
            None,
            remaining_words=["APPLE"],
            feedback="",
            rag_context="",
            validator_feedback="",
            previous_understanding=[["A", "B", "C", "D"], ["E", "F", "G", "H"]],
        )
        assert "Last Board Understanding" in prompt
        assert "A, B, C, D" in prompt

    def test_with_validator_feedback(self):
        prompt = GuesserAgent._build_prompt(
            None,
            remaining_words=["APPLE"],
            feedback="",
            rag_context="",
            validator_feedback="Try a different group",
            previous_understanding=None,
        )
        assert "Validator Feedback" in prompt
        assert "Try a different group" in prompt

    def test_with_rag_context(self):
        prompt = GuesserAgent._build_prompt(
            None,
            remaining_words=["APPLE"],
            feedback="",
            rag_context="Historical puzzle: FRUITS",
            validator_feedback="",
            previous_understanding=None,
        )
        assert "Historical Context" in prompt
        assert "Historical puzzle: FRUITS" in prompt

    def test_section_ordering(self):
        """Sections should appear in order: feedback, understanding, validator, rag, words."""
        prompt = GuesserAgent._build_prompt(
            None,
            remaining_words=["APPLE"],
            feedback="FB",
            rag_context="RAG",
            validator_feedback="VAL",
            previous_understanding=[["X", "Y", "Z", "W"]],
        )
        fb_pos = prompt.index("FB")
        und_pos = prompt.index("Last Board Understanding")
        val_pos = prompt.index("VAL")
        rag_pos = prompt.index("RAG")
        words_pos = prompt.index("Remaining Words")
        assert fb_pos < und_pos < val_pos < rag_pos < words_pos


class TestGuesserSystemPrompt:
    """Test that the system prompt includes cross-category reasoning instructions."""

    def test_cross_category_instructions(self):
        from gvc_local.agents.guesser import GUESSER_SYSTEM_PROMPT

        assert "ALL 4 GROUPS SIMULTANEOUSLY" in GUESSER_SYSTEM_PROMPT
        assert "PROCESS OF ELIMINATION" in GUESSER_SYSTEM_PROMPT
        assert "ONE AWAY" in GUESSER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SnapGuesserAgent tests
# ---------------------------------------------------------------------------


class TestSnapGuesserParsing:
    def test_clean_json(self):
        reply = '{"reason": "All fruits", "words": ["APPLE", "BANANA", "CHERRY", "DATE"]}'
        group, reason = SnapGuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert reason == "All fruits"

    def test_json_with_code_fence(self):
        reply = """```json
{"reason": "All fruits", "words": ["APPLE", "BANANA", "CHERRY", "DATE"]}
```"""
        group, reason = SnapGuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]

    def test_regex_fallback(self):
        reply = """Here is my guess:
"reason": "All fruits", "words": ["APPLE", "BANANA", "CHERRY", "DATE"]
"""
        group, reason = SnapGuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert reason == "All fruits"

    def test_missing_words_raises(self):
        reply = '{"reason": "something"}'
        with pytest.raises(ValueError, match="words"):
            SnapGuesserAgent._parse_reply(reply)

    def test_wrong_word_count_raises(self):
        reply = '{"reason": "x", "words": ["A", "B", "C"]}'
        with pytest.raises(ValueError, match="Expected 4"):
            SnapGuesserAgent._parse_reply(reply)

    def test_missing_reason_ok(self):
        reply = '{"words": ["APPLE", "BANANA", "CHERRY", "DATE"]}'
        group, reason = SnapGuesserAgent._parse_reply(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]
        assert reason == ""


class TestSnapGuesserPrompt:
    def test_basic_prompt(self):
        prompt = SnapGuesserAgent._build_prompt(["A", "B", "C", "D"], "")
        assert "A, B, C, D" in prompt
        assert "4 words" in prompt

    def test_prompt_with_feedback(self):
        prompt = SnapGuesserAgent._build_prompt(["A"], "Don't guess X")
        assert "Don't guess X" in prompt


# ---------------------------------------------------------------------------
# ValidatorAgent tests
# ---------------------------------------------------------------------------


class TestValidatorParsing:
    def test_agreement_true(self):
        reply = """Agreement to Perform the Guess: True
Feedback for Guesser Agent: The group looks correct."""
        agreement, feedback = ValidatorAgent._parse_reply(reply)
        assert agreement is True
        assert "looks correct" in feedback

    def test_agreement_false(self):
        reply = """Agreement to Perform the Guess: False
Feedback for Guesser Agent: WORD1 doesn't fit the category."""
        agreement, feedback = ValidatorAgent._parse_reply(reply)
        assert agreement is False
        assert "doesn't fit" in feedback

    def test_fallback_agree(self):
        reply = "I agree with the guesser's proposal."
        agreement, feedback = ValidatorAgent._parse_reply(reply)
        assert agreement is True

    def test_fallback_disagree(self):
        reply = "I disagree. The grouping is wrong because of X."
        agreement, feedback = ValidatorAgent._parse_reply(reply)
        assert agreement is False

    def test_no_feedback_text(self):
        reply = "Agreement to Perform the Guess: True"
        agreement, feedback = ValidatorAgent._parse_reply(reply)
        assert agreement is True
        assert feedback == ""


class TestValidatorExtractGroup:
    def test_extract_group(self):
        reply = "Group: APPLE, BANANA, CHERRY, DATE"
        group = ValidatorAgent._extract_group(reply)
        assert group == ["APPLE", "BANANA", "CHERRY", "DATE"]

    def test_extract_group_wrong_count(self):
        reply = "Group: APPLE, BANANA, CHERRY"
        group = ValidatorAgent._extract_group(reply)
        assert group == []

    def test_extract_group_no_match(self):
        reply = "The guesser's proposal is wrong."
        group = ValidatorAgent._extract_group(reply)
        assert group == []


class TestValidatorPrompt:
    def test_build_prompt(self):
        prompt = ValidatorAgent._build_prompt(
            guesser_reply="Group: A, B, C, D\nCategory: TEST",
            remaining_words=["A", "B", "C", "D", "E", "F", "G", "H"],
            feedback="",
        )
        assert "Guesser Agent's reply" in prompt
        assert "A, B, C, D, E, F, G, H" in prompt

    def test_build_prompt_with_feedback(self):
        prompt = ValidatorAgent._build_prompt(
            guesser_reply="test",
            remaining_words=["A"],
            feedback="Failed: X, Y, Z, W",
        )
        assert "Game Engine Feedback" in prompt
        assert "Failed: X, Y, Z, W" in prompt
