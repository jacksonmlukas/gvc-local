"""Validator agent -- reviews the guesser's proposed group and category.

The Validator receives the guesser's full reply (including its board
understanding and chosen group + category), along with the remaining words
and game-engine feedback. It decides whether to **agree** with the guess
or **reject** it with feedback explaining why.

Expected LLM output format::

    Agreement to Perform the Guess: True
    Feedback for Guesser Agent: Looks good, the group is coherent.

or::

    Agreement to Perform the Guess: False
    Feedback for Guesser Agent: WORD_X does not fit the category because ...
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

VALIDATOR_SYSTEM_PROMPT = """\
You are an Expert Word Grouping Validator for the NYT Connections puzzle.
You understand literature, culture, and are well-versed in common phrases \
and wordplay.

You will receive:
1. The Guesser Agent's full reply (including their board understanding and \
   chosen group + category).
2. The remaining words on the board.
3. Game Engine feedback about previously failed guesses.

Your job:
- Evaluate whether the guesser's chosen 4-word group genuinely fits the \
  proposed category better than any alternative 4-word group from the \
  remaining words.
- Check that all 4 words are actually on the board.
- Check that the group has not been previously tried and failed.

Format your response EXACTLY as follows:

Agreement to Perform the Guess: True
Feedback for Guesser Agent: <your reasoning>

or:

Agreement to Perform the Guess: False
Feedback for Guesser Agent: <explain why the guess should be rejected and \
suggest improvements>
"""


class ValidatorAgent(Agent):
    """Reviews a guesser's proposed group and decides whether to approve it.

    Parameters
    ----------
    client:
        The vLLM endpoint client.
    temperature:
        Default temperature for the validator (upstream uses 0.7--0.8).
    """

    def __init__(self, client: Client, temperature: float = 0.7) -> None:
        super().__init__(client, role="Validator", system_prompt=VALIDATOR_SYSTEM_PROMPT)
        self.default_temperature = temperature

    def validate(
        self,
        guesser_reply: str,
        remaining_words: list[str],
        feedback: str = "",
    ) -> tuple[bool, str, list[str]]:
        """Validate a guesser's proposed guess.

        Parameters
        ----------
        guesser_reply:
            The full raw reply from the guesser agent.
        remaining_words:
            Words still on the board.
        feedback:
            Game-engine feedback about previously failed guesses.

        Returns
        -------
        A 3-tuple of:
            - ``agreement``: whether the validator agrees with the guess.
            - ``feedback_text``: the validator's written feedback / reasoning.
            - ``validator_group``: the 4 words the validator thinks should be
              guessed (may differ from the guesser's if it disagrees). Empty
              list if no alternative group was extracted.
        """
        prompt = self._build_prompt(guesser_reply, remaining_words, feedback)

        reply = self.respond(prompt, temperature=self.default_temperature)
        logger.info("[Validator] raw reply:\n%s", reply)

        agreement, feedback_text = self._parse_reply(reply)
        # Try to extract an alternative group if the validator mentions one
        validator_group = self._extract_group(reply)

        logger.info(
            "[Validator] agreement=%s  feedback=%s  alt_group=%s",
            agreement,
            feedback_text[:80] if feedback_text else "",
            validator_group,
        )
        return agreement, feedback_text, validator_group

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        guesser_reply: str,
        remaining_words: list[str],
        feedback: str,
    ) -> str:
        sections: list[str] = []

        sections.append(
            f"**Context:**\n"
            f'Guesser Agent\'s reply START: """\n{guesser_reply}\n"""\n'
            f"Guesser Agent's reply END"
        )

        remaining_str = ", ".join(remaining_words)
        sections.append(f"**Remaining Words:**\nWords left on the board: {remaining_str}")

        if feedback:
            sections.append(f"**Game Engine Feedback**\n{feedback}")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reply(reply: str) -> tuple[bool, str]:
        """Parse the validator's reply for agreement and feedback.

        Returns (agreement_bool, feedback_text).
        """
        # Agreement
        agree_m = re.search(
            r"Agreement to Perform the Guess:\s*(True|False)",
            reply,
            re.IGNORECASE,
        )
        if agree_m:
            agreement = agree_m.group(1).strip().lower() == "true"
        else:
            # Heuristic fallback: if "agree" or "yes" appears prominently
            lower = reply.lower()
            if "consensus reached" in lower or "i agree" in lower:
                agreement = True
            else:
                agreement = False
                logger.warning(
                    "[Validator] could not parse agreement from reply, defaulting to False"
                )

        # Feedback text
        fb_m = re.search(
            r"Feedback for Guesser Agent:\s*(.*?)(?:\n<|$)",
            reply,
            re.DOTALL,
        )
        feedback_text = fb_m.group(1).strip() if fb_m else ""

        return agreement, feedback_text

    @staticmethod
    def _extract_group(reply: str) -> list[str]:
        """Try to extract a 'Group: w1, w2, w3, w4' line from the reply."""
        m = re.search(r"Group:\s*(.+)", reply, re.IGNORECASE)
        if m:
            words = [
                w.strip().upper().replace(",", "")
                for w in re.split(r",\s*", m.group(1))
                if w.strip()
            ]
            if len(words) == 4:
                return words
        return []
