"""Thin adapter that re-exports the upstream game module.

Tries to import from the upstream ``rsallms.game`` package first. If the
upstream repo is not installed (e.g. in a standalone deployment), falls back
to a minimal local implementation that provides the same public API:

    Connections, Category, GameOverError, load_games, sample_game
"""

from __future__ import annotations

try:
    # Prefer the upstream implementation when available on sys.path.
    from rsallms.game import (  # type: ignore[import-untyped]
        Category,
        Connections,
        GameOverError,
        load_games,
        load_json_to_connections,
        sample_game,
    )
except ImportError:
    # -----------------------------------------------------------------
    # Standalone fallback -- mirrors the upstream API surface.
    # -----------------------------------------------------------------
    import json
    import random
    from dataclasses import dataclass

    import requests

    GAME_DATA_ENDPOINT = (
        "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/"
        "refs/heads/main/connections.json"
    )

    class GameOverError(Exception):  # type: ignore[no-redef]
        """Raised when the maximum number of strikes has been reached."""

    @dataclass
    class Category:  # type: ignore[no-redef]
        """A single category (group) in a Connections puzzle."""

        level: int
        group: str
        members: list[str]

        def matches(self, words: list[str]) -> bool:
            return set(words) == set(self.members)

        def diff(self, other_category: Category) -> int:
            return len(set(self.members).symmetric_difference(set(other_category.members)))

    class Connections:  # type: ignore[no-redef]
        """A single game of Connections."""

        def __init__(
            self,
            categories: list[Category],
            group_size: int = 4,
            max_strikes: int = 20,
            starting_strikes: int = 0,
        ) -> None:
            if not all(len(g.members) == group_size for g in categories):
                raise ValueError(f"All groups must have exactly {group_size} members")
            self._max_strikes = max_strikes
            self._og_groups: list[Category] = categories.copy()
            self.group_size = group_size
            self.categories: list[Category] = categories.copy()
            self.current_strikes = starting_strikes

        # -- properties ------------------------------------------------

        @property
        def all_words(self) -> list[str]:
            words = [w for g in self.categories for w in g.members]
            random.shuffle(words)
            return words

        @property
        def is_solved(self) -> bool:
            return len(self.categories) == 0

        @property
        def is_over(self) -> bool:
            return self.current_strikes >= self._max_strikes or self.is_solved

        @property
        def solved_categories(self) -> list[bool]:
            return [cat not in self.categories for cat in self._og_groups]

        # -- game logic ------------------------------------------------

        def category_guess_check(self, words: list[str]) -> Category | None:
            if self.current_strikes >= self._max_strikes:
                raise GameOverError("Max strikes reached!")
            for i, group in enumerate(self.categories):
                if group.matches(words):
                    self._last_one_away = False
                    return self.categories.pop(i)
            self.current_strikes += 1
            # Check if the guess was "one away" from any remaining category
            self._last_one_away = any(
                len(set(words) & set(cat.members)) == 3 for cat in self.categories
            )
            return None

        @property
        def last_one_away(self) -> bool:
            """True if the most recent wrong guess had 3/4 words correct."""
            return getattr(self, "_last_one_away", False)

        def reset(self) -> None:
            self.categories = self._og_groups.copy()
            self.current_strikes = 0

        def __str__(self) -> str:
            lines = [
                f"Strikes: {self.current_strikes}/{self._max_strikes}  "
                f"Solved: {sum(self.solved_categories)}/{len(self._og_groups)}"
            ]
            for cat in self._og_groups:
                solved = cat not in self.categories
                lines.append(
                    f"  {'[x]' if solved else '[ ]'} {cat.group}: {', '.join(cat.members)}"
                )
            return "\n".join(lines)

    def load_games() -> list[Connections]:  # type: ignore[no-redef]
        """Fetch all historical puzzles from the public GitHub data source."""
        resp = requests.get(GAME_DATA_ENDPOINT, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON list of games")
        return [
            Connections(categories=[Category(**cat) for cat in game["answers"]]) for game in raw
        ]

    def load_json_to_connections(filename: str) -> list[Connections]:  # type: ignore[no-redef]
        """Load puzzle data from a local JSON file."""
        with open(filename) as f:
            data = json.load(f)
        categories = [Category(**item) for item in data]
        return [Connections(categories[i : i + 4]) for i in range(0, len(categories), 4)]

    def sample_game() -> Connections:  # type: ignore[no-redef]
        """Return a randomly sampled historical game."""
        return random.choice(load_games())


__all__ = [
    "Category",
    "Connections",
    "GameOverError",
    "load_games",
    "load_json_to_connections",
    "sample_game",
]
