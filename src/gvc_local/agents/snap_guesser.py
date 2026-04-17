"""Snap (System-1) guesser agent -- fast, high-temperature intuitive guessing.

When the conservative GVC loop gets stuck, the Snap agent takes over with a
high temperature (0.9) and a simpler prompt to produce quick, intuition-driven
guesses. This is the *System-1* component of the dual-process architecture
described in the "Snap Out of It" paper.

Expected LLM output format (JSON)::

    {"reason": "These are all types of dance", "words": ["WALTZ", "SALSA", "TANGO", "SWING"]}
"""

from __future__ import annotations

import json
import logging
import re

from gvc_local.agents.base import Agent
from gvc_local.endpoint import Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SNAP_SYSTEM_PROMPT = """\
You are a fast, intuitive word-grouping agent for the NYT Connections puzzle.
Trust your gut. Given a set of words, quickly identify one group of exactly \
4 words that share a strong connection.

You MUST respond with ONLY a JSON object in this exact format:
{"reason": "<short explanation>", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]}

Do not include any other text, explanation, or formatting outside the JSON object.
"""


class SnapGuesserAgent(Agent):
    """Fast, intuition-driven guesser (System-1).

    Uses a higher temperature than the deliberative guesser to encourage
    creative / lateral connections when the conservative loop is stuck.

    Parameters
    ----------
    client:
        The vLLM endpoint client.
    temperature:
        Default temperature (paper uses 0.9 for snap guesses).
    """

    def __init__(self, client: Client, temperature: float = 0.9) -> None:
        super().__init__(client, role="SnapGuesser", system_prompt=SNAP_SYSTEM_PROMPT)
        self.default_temperature = temperature

    def snap_guess(
        self,
        remaining_words: list[str],
        feedback: str = "",
    ) -> tuple[list[str], str]:
        """Produce a quick intuitive guess.

        Parameters
        ----------
        remaining_words:
            Words still on the board.
        feedback:
            Accumulated game-engine feedback about previously failed guesses.

        Returns
        -------
        A 2-tuple of:
            - ``group``: list of 4 words
            - ``reason``: short explanation / category-like label

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into the expected format.
        """
        prompt = self._build_prompt(remaining_words, feedback)

        reply = self.respond(prompt, temperature=self.default_temperature)
        logger.info("[SnapGuesser] raw reply:\n%s", reply)

        group, reason = self._parse_reply(reply)

        # Normalise
        group = [w.strip().upper().replace(",", "") for w in group]

        logger.info("[SnapGuesser] group=%s  reason=%s", group, reason)
        return group, reason

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(remaining_words: list[str], feedback: str) -> str:
        remaining_str = ", ".join(remaining_words)
        parts: list[str] = [f"Here are some words: {remaining_str}"]

        if feedback:
            parts.append(feedback)

        parts.append("Task: Create one logical grouping that uses 4 words.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reply(reply: str) -> tuple[list[str], str]:
        """Parse the snap guesser's JSON reply.

        Handles both clean JSON and JSON embedded in markdown code fences.
        Falls back to regex extraction if ``json.loads`` fails.
        """
        # Strip markdown fences if present
        cleaned = re.sub(r"```json?\s*", "", reply)
        cleaned = re.sub(r"```", "", cleaned).strip()

        # Try direct JSON parse first
        try:
            obj = json.loads(cleaned)
            words = obj["words"]
            reason = obj.get("reason", "")
            if isinstance(words, list) and len(words) == 4:
                return [str(w) for w in words], str(reason)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Regex fallback -- mirrors upstream parse_snap_guesser_reply
        reason_m = re.search(r'"reason"\s*:\s*"(.*?)"', reply)
        words_m = re.search(r'"words"\s*:\s*\[(.*?)\]', reply)

        if not words_m:
            raise ValueError(f"Could not parse 'words' from snap reply:\n{reply[:500]}")

        reason = reason_m.group(1) if reason_m else ""
        words_raw = words_m.group(1)
        words = [w.strip().strip('"').strip("'") for w in words_raw.split(",")]

        if len(words) != 4:
            raise ValueError(
                f"Expected 4 words in snap guess, got {len(words)}: {words}"
            )

        return words, reason
