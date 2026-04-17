"""Guesser agent -- proposes a group of 4 words and a category label.

The Guesser is the *System-2* deliberative component of the GVC loop. It
receives the remaining board words, accumulated feedback from failed guesses
and the validator, and optional RAG context. It produces structured output
that gets parsed into (group, category, board_understanding).

Output format expected from the LLM::

    <UNDERSTANDING_OF_BOARD>
    Group1: WORD1, WORD2, WORD3, WORD4
    Group2: WORD5, WORD6, WORD7, WORD8
    ...
    <END_UNDERSTANDING_OF_BOARD>

    <GUESS_FOR_THIS_ROUND>
    Group: WORD_A, WORD_B, WORD_C, WORD_D
    Category: SOME CATEGORY NAME
    <END_GUESS_FOR_THIS_ROUND>
"""

from __future__ import annotations

import logging
import re

from gvc_local.agents.base import Agent
from gvc_local.endpoint import Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

GUESSER_SYSTEM_PROMPT = """\
You are an Expert Word Grouping Agent playing the NYT Connections puzzle.
You deeply understand literature, culture, and are well-versed in common \
phrases and wordplay. You know every definition of every word. You understand \
how to create fill-in-the-blank category names.

Given a list of words, propose exactly 4 groups of 4 related words and their \
corresponding categories. Your categories should be specific enough that \
another agent could distinguish the group of four words from the word bank \
solely based on the category name.

**DO NOT GUESS A PREVIOUSLY GUESSED CATEGORY.**

Refer to these category style examples for guidance:
CONTORTED, CUT THE ___, KINDS OF PICKLES, ESCAPADE, PUBLIC STANDING, \
GROUNDBREAKING, THINGS WITH SHELLS, INDIVIDUALITY, WORDS WITH APOSTROPHES \
REMOVED, EQUIP, EASY ___, LEGAL SESSION, HEARTWARMING, CORE EXERCISES, \
SNEAKER BRANDS, MUSICALS BEGINNING WITH "C", CLEANING VERBS, ___ MAN \
SUPERHEROES, STREAMING SERVICES, CONDIMENTS, SYNONYMS FOR SAD, CLUE \
CHARACTERS, MONOPOLY SQUARES, SHADES OF BLUE, RAPPERS, MEMBERS OF A SEPTET, \
LEG PARTS, BABY ANIMALS, SLANG FOR TOILET, ___ FISH THAT AREN'T FISH

You MUST format your response EXACTLY as follows (including the XML-like tags):

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

Pick the group you are MOST confident about as your guess for this round.
"""


class GuesserAgent(Agent):
    """Proposes a group of 4 words and a category from the remaining board.

    Parameters
    ----------
    client:
        The vLLM endpoint client.
    temperature:
        Default temperature for the guesser (upstream uses 0.7--1.1).
    """

    def __init__(self, client: Client, temperature: float = 0.7) -> None:
        super().__init__(client, role="Guesser", system_prompt=GUESSER_SYSTEM_PROMPT)
        self.default_temperature = temperature

    def guess(
        self,
        remaining_words: list[str],
        feedback: str = "",
        rag_context: str = "",
        validator_feedback: str = "",
        previous_understanding: list[list[str]] | None = None,
    ) -> tuple[list[str], str, list[list[str]]]:
        """Generate a guess from the remaining words.

        Parameters
        ----------
        remaining_words:
            Words still on the board.
        feedback:
            Accumulated game-engine feedback about failed guesses.
        rag_context:
            Formatted RAG retrieval context (from ``RetrievalResult.format_for_prompt``).
        validator_feedback:
            Text feedback from the validator on the last rejected internal guess.
        previous_understanding:
            The guesser's last board understanding (list of word-groups).

        Returns
        -------
        A 3-tuple of:
            - ``group``: list of 4 words for this guess
            - ``category``: the proposed category label
            - ``board_understanding``: list of lists representing how the
              guesser currently partitions the board
        """
        prompt = self._build_prompt(
            remaining_words,
            feedback=feedback,
            rag_context=rag_context,
            validator_feedback=validator_feedback,
            previous_understanding=previous_understanding,
        )

        reply = self.respond(prompt, temperature=self.default_temperature)
        logger.info("[Guesser] raw reply:\n%s", reply)

        group, category, understanding = self._parse_reply(reply)

        # Normalise words to uppercase, strip whitespace
        group = [w.strip().upper().replace(",", "") for w in group]
        understanding = [
            [w.strip().upper().replace(",", "") for w in grp]
            for grp in understanding
        ]

        logger.info("[Guesser] group=%s  category=%s", group, category)
        return group, category, understanding

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        remaining_words: list[str],
        *,
        feedback: str,
        rag_context: str,
        validator_feedback: str,
        previous_understanding: list[list[str]] | None,
    ) -> str:
        sections: list[str] = []

        if feedback:
            sections.append(f"**Game Engine Feedback**\n{feedback}")

        if previous_understanding:
            lines = "\n".join(
                f"  * {', '.join(grp)}" for grp in previous_understanding
            )
            sections.append(
                f"**Your Last Board Understanding**\n"
                f"- This is your previous understanding of the board:\n{lines}"
            )

        if validator_feedback:
            sections.append(f"**Validator Feedback**\n{validator_feedback}")

        if rag_context:
            sections.append(f"**Historical Context**\n{rag_context}")

        remaining_str = ", ".join(remaining_words)
        sections.append(f"**Remaining Words**\nWords: {remaining_str}")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reply(reply: str) -> tuple[list[str], str, list[list[str]]]:
        """Parse the guesser's structured reply.

        Returns (group, category, board_understanding).

        Raises ``ValueError`` when the expected format is not found.
        """
        # Normalise whitespace while keeping newlines meaningful
        normalized = re.sub(r"[ \t]*\n[ \t]*", "\n", reply.strip())

        # --- Board understanding ---
        understanding: list[list[str]] = []
        und_match = re.search(
            r"<UNDERSTANDING_OF_BOARD>(.*?)<END_UNDERSTANDING_OF_BOARD>",
            normalized,
            re.DOTALL,
        )
        if und_match:
            und_text = und_match.group(1)
            # Match lines like "Group1: WORD, WORD, WORD, WORD" or "Group 1: ..."
            group_lines = re.findall(
                r"Group\s*\d+\s*:\s*(.+)", und_text, re.IGNORECASE
            )
            for line in group_lines:
                words = [w.strip() for w in re.split(r",\s*", line) if w.strip()]
                if words:
                    understanding.append(words)

        # --- Guess for this round ---
        guess_match = re.search(
            r"<GUESS_FOR_THIS_ROUND>(.*?)<END_GUESS_FOR_THIS_ROUND>",
            normalized,
            re.DOTALL,
        )
        if guess_match:
            guess_text = guess_match.group(1).strip()
        else:
            # Fallback: look for Group:/Category: lines anywhere
            guess_text = normalized

        group_m = re.search(r"Group:\s*(.+)", guess_text, re.IGNORECASE)
        cat_m = re.search(r"Category:\s*(.+)", guess_text, re.IGNORECASE)

        if not group_m:
            raise ValueError(f"Could not find 'Group:' in guesser reply:\n{reply[:500]}")
        if not cat_m:
            raise ValueError(f"Could not find 'Category:' in guesser reply:\n{reply[:500]}")

        group = [w.strip() for w in re.split(r",\s*", group_m.group(1)) if w.strip()]
        category = cat_m.group(1).strip()

        if len(group) != 4:
            raise ValueError(
                f"Expected 4 words in group, got {len(group)}: {group}"
            )

        return group, category, understanding
