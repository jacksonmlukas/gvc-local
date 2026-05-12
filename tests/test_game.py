"""Tests for the game engine — Category, Connections, one-away detection."""

import pytest

from gvc_local.game import Category, Connections, GameOverError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_categories() -> list[Category]:
    """Build four test categories for a standard puzzle."""
    return [
        Category(level=1, group="FRUITS", members=["APPLE", "BANANA", "CHERRY", "DATE"]),
        Category(level=2, group="COLORS", members=["RED", "BLUE", "GREEN", "YELLOW"]),
        Category(level=3, group="ANIMALS", members=["CAT", "DOG", "FISH", "BIRD"]),
        Category(level=4, group="NUMBERS", members=["ONE", "TWO", "THREE", "FOUR"]),
    ]


def _make_game(**kwargs) -> Connections:
    return Connections(categories=_make_categories(), **kwargs)


# ---------------------------------------------------------------------------
# Category tests
# ---------------------------------------------------------------------------


class TestCategory:
    def test_matches_exact(self):
        cat = Category(level=1, group="FRUITS", members=["APPLE", "BANANA", "CHERRY", "DATE"])
        assert cat.matches(["APPLE", "BANANA", "CHERRY", "DATE"])

    def test_matches_different_order(self):
        cat = Category(level=1, group="FRUITS", members=["APPLE", "BANANA", "CHERRY", "DATE"])
        assert cat.matches(["DATE", "CHERRY", "BANANA", "APPLE"])

    def test_no_match_wrong_words(self):
        cat = Category(level=1, group="FRUITS", members=["APPLE", "BANANA", "CHERRY", "DATE"])
        assert not cat.matches(["APPLE", "BANANA", "CHERRY", "GRAPE"])

    def test_diff_identical(self):
        cat1 = Category(level=1, group="A", members=["X", "Y", "Z", "W"])
        cat2 = Category(level=1, group="B", members=["X", "Y", "Z", "W"])
        assert cat1.diff(cat2) == 0

    def test_diff_one_different(self):
        cat1 = Category(level=1, group="A", members=["X", "Y", "Z", "W"])
        cat2 = Category(level=1, group="B", members=["X", "Y", "Z", "Q"])
        assert cat1.diff(cat2) == 2  # symmetric difference: {W, Q}

    def test_diff_all_different(self):
        cat1 = Category(level=1, group="A", members=["A", "B", "C", "D"])
        cat2 = Category(level=1, group="B", members=["W", "X", "Y", "Z"])
        assert cat1.diff(cat2) == 8


# ---------------------------------------------------------------------------
# Connections game tests
# ---------------------------------------------------------------------------


class TestConnections:
    def test_initial_state(self):
        game = _make_game()
        assert not game.is_solved
        assert not game.is_over
        assert game.current_strikes == 0
        assert len(game.all_words) == 16

    def test_all_words_contains_all(self):
        game = _make_game()
        words = game.all_words
        assert set(words) == {
            "APPLE", "BANANA", "CHERRY", "DATE",
            "RED", "BLUE", "GREEN", "YELLOW",
            "CAT", "DOG", "FISH", "BIRD",
            "ONE", "TWO", "THREE", "FOUR",
        }

    def test_correct_guess_removes_category(self):
        game = _make_game()
        result = game.category_guess_check(["APPLE", "BANANA", "CHERRY", "DATE"])
        assert result is not None
        assert result.group == "FRUITS"
        assert len(game.categories) == 3
        assert len(game.all_words) == 12

    def test_wrong_guess_increments_strikes(self):
        game = _make_game()
        result = game.category_guess_check(["APPLE", "BANANA", "RED", "CAT"])
        assert result is None
        assert game.current_strikes == 1

    def test_solved_after_all_correct(self):
        game = _make_game()
        game.category_guess_check(["APPLE", "BANANA", "CHERRY", "DATE"])
        game.category_guess_check(["RED", "BLUE", "GREEN", "YELLOW"])
        game.category_guess_check(["CAT", "DOG", "FISH", "BIRD"])
        game.category_guess_check(["ONE", "TWO", "THREE", "FOUR"])
        assert game.is_solved
        assert game.is_over
        assert game.solved_categories == [True, True, True, True]

    def test_game_over_at_max_strikes(self):
        game = _make_game(max_strikes=3)
        wrong = ["APPLE", "RED", "CAT", "ONE"]
        game.category_guess_check(wrong)
        game.category_guess_check(wrong)
        game.category_guess_check(wrong)
        assert game.is_over
        with pytest.raises(GameOverError):
            game.category_guess_check(wrong)

    def test_starting_strikes(self):
        game = _make_game(starting_strikes=5, max_strikes=10)
        assert game.current_strikes == 5

    def test_solved_categories_partial(self):
        game = _make_game()
        game.category_guess_check(["RED", "BLUE", "GREEN", "YELLOW"])
        solved = game.solved_categories
        assert solved == [False, True, False, False]

    def test_reset(self):
        game = _make_game()
        game.category_guess_check(["APPLE", "BANANA", "CHERRY", "DATE"])
        game.category_guess_check(["APPLE", "RED", "CAT", "ONE"])  # wrong
        game.reset()
        assert game.current_strikes == 0
        assert len(game.categories) == 4

    def test_str_representation(self):
        game = _make_game()
        s = str(game)
        assert "FRUITS" in s
        assert "Strikes: 0" in s

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="exactly"):
            Connections(categories=[
                Category(level=1, group="BAD", members=["A", "B", "C"]),
            ], group_size=4)


# ---------------------------------------------------------------------------
# One-away detection tests
# ---------------------------------------------------------------------------


class TestOneAway:
    def test_one_away_true(self):
        """Guess with 3/4 correct words should trigger one_away."""
        game = _make_game()
        # 3 fruits + 1 wrong
        result = game.category_guess_check(["APPLE", "BANANA", "CHERRY", "RED"])
        assert result is None
        assert game.last_one_away is True

    def test_one_away_false_two_wrong(self):
        """Guess with 2/4 correct should NOT trigger one_away."""
        game = _make_game()
        result = game.category_guess_check(["APPLE", "BANANA", "RED", "BLUE"])
        assert result is None
        assert game.last_one_away is False

    def test_one_away_false_all_wrong(self):
        """Guess from different categories should NOT trigger one_away."""
        game = _make_game()
        result = game.category_guess_check(["APPLE", "RED", "CAT", "ONE"])
        assert result is None
        assert game.last_one_away is False

    def test_one_away_reset_on_correct(self):
        """After a correct guess, one_away should be False."""
        game = _make_game()
        # First: one-away wrong guess
        game.category_guess_check(["APPLE", "BANANA", "CHERRY", "RED"])
        assert game.last_one_away is True
        # Then: correct guess
        game.category_guess_check(["APPLE", "BANANA", "CHERRY", "DATE"])
        assert game.last_one_away is False

    def test_one_away_default(self):
        """Fresh game should have last_one_away=False."""
        game = _make_game()
        assert game.last_one_away is False
