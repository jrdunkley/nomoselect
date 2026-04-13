"""Tests for nomoselect.ladder — DimensionCostLadder."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine

from nomoselect import DimensionCostLadder


@pytest.fixture
def iris():
    d = load_iris()
    return d.data, d.target


@pytest.fixture
def wine():
    d = load_wine()
    return d.data, d.target


class TestLadderBuild:
    def test_basic_build(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        assert len(ladder.rungs) == 2
        assert ladder.rungs[0].rank == 1
        assert ladder.rungs[1].rank == 2

    def test_winner_at_zero(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        # Winner at c=0 is the highest score, which should be rank 2
        assert len(ladder.winner_at_zero) >= 1
        assert all(w in [1, 2] for w in ladder.winner_at_zero)

    def test_rungs_have_diagnostics(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        for rung in ladder.rungs:
            assert rung.visible_score >= 0
            assert 0 <= rung.eta <= 1.0 + 1e-10
            assert isinstance(rung.is_viable, bool)

    def test_recommended_rank_present(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        assert ladder.recommended_rank in [1, 2]

    def test_wine_2_ranks(self, wine):
        X, y = wine
        # Wine has 3 classes → max useful rank = 2
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        assert len(ladder.rungs) == 2

    def test_summary_string(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        s = ladder.summary()
        assert "ladder" in s.lower()
        assert "recommended" in s.lower() or "Recommended" in s
        # Technical mode
        s_tech = ladder.summary(technical=True)
        assert "DeclaredLadder" in s_tech

    def test_default_max_rank(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y)
        # Iris has 3 classes → default max_rank = min(k-1, 10) = 2
        assert len(ladder.rungs) == 2


class TestLadderMonotonicity:
    """Visible score should generally increase with rank."""

    def test_score_nondecreasing(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        scores = [r.visible_score for r in ladder.rungs]
        # Rank 2 should capture at least as much as rank 1
        assert scores[1] >= scores[0] - 1e-10
