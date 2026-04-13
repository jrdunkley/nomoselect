"""
Hostile-user tests — verify nomoselect fails gracefully when misused.

These are not correctness tests.  They check that bad inputs produce
clear error messages rather than cryptic numpy exceptions.
"""
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris

from nomoselect import (
    GeometricSubspaceSelector,
    ObserverReport,
    RegularisationAudit,
    DimensionCostLadder,
)


@pytest.fixture
def iris():
    d = load_iris()
    return d.data, d.target


# ── Rank too high ──────────────────────────────────────────────────


class TestRankTooHigh:
    def test_rank_exceeds_k_minus_1(self, iris):
        X, y = iris  # 3 classes → max useful rank = 2
        with pytest.raises(ValueError, match="n_components=5 is too large"):
            GeometricSubspaceSelector(n_components=5).fit(X, y)

    def test_rank_equals_k(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="n_components=3 is too large"):
            GeometricSubspaceSelector(n_components=3).fit(X, y)

    def test_rank_zero(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="at least 1"):
            GeometricSubspaceSelector(n_components=0).fit(X, y)

    def test_rank_negative(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="at least 1"):
            GeometricSubspaceSelector(n_components=-1).fit(X, y)

    def test_rank_float(self, iris):
        X, y = iris
        with pytest.raises(TypeError, match="integer"):
            GeometricSubspaceSelector(n_components=1.5).fit(X, y)


# ── Bad labels ─────────────────────────────────────────────────────


class TestBadLabels:
    def test_single_class(self, iris):
        X, y = iris
        y_single = np.zeros(len(y), dtype=int)
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            GeometricSubspaceSelector(n_components=1).fit(X, y_single)

    def test_all_same_label(self, iris):
        X, y = iris
        y_same = np.full(len(y), "cat")
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            GeometricSubspaceSelector(n_components=1).fit(X, y_same)

    def test_class_with_one_sample(self, iris):
        X, y = iris
        # Make class 2 have exactly 1 sample
        mask = y != 2
        X_sub = np.vstack([X[mask], X[y == 2][:1]])
        y_sub = np.hstack([y[mask], [2]])
        with pytest.raises(ValueError, match="only 1 sample"):
            GeometricSubspaceSelector(n_components=1).fit(X_sub, y_sub)


# ── Bad data ───────────────────────────────────────────────────────


class TestBadData:
    def test_nan_in_X(self, iris):
        X, y = iris
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GeometricSubspaceSelector(n_components=1).fit(X_bad, y)

    def test_inf_in_X(self, iris):
        X, y = iris
        X_bad = X.copy()
        X_bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            GeometricSubspaceSelector(n_components=1).fit(X_bad, y)

    def test_1d_X(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="2-D"):
            GeometricSubspaceSelector(n_components=1).fit(X[:, 0], y)

    def test_3d_X(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="2-D"):
            GeometricSubspaceSelector(n_components=1).fit(
                X.reshape(150, 2, 2), y
            )

    def test_X_y_length_mismatch(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="1-D array with 150"):
            GeometricSubspaceSelector(n_components=1).fit(X, y[:100])

    def test_string_X(self, iris):
        _, y = iris
        X_str = np.array([["a", "b"]] * len(y))
        with pytest.raises(ValueError, match="numeric"):
            GeometricSubspaceSelector(n_components=1).fit(X_str, y)

    def test_one_sample(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            GeometricSubspaceSelector(n_components=1).fit(
                np.array([[1.0, 2.0]]), np.array([0])
            )


# ── Bad parameters ─────────────────────────────────────────────────


class TestBadParameters:
    def test_reg_floor_negative(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            GeometricSubspaceSelector(
                n_components=1, reg_floor_frac=-0.01
            ).fit(X, y)

    def test_reg_floor_one(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            GeometricSubspaceSelector(
                n_components=1, reg_floor_frac=1.0
            ).fit(X, y)

    def test_reg_floor_string(self, iris):
        X, y = iris
        with pytest.raises(TypeError, match="number"):
            GeometricSubspaceSelector(
                n_components=1, reg_floor_frac="high"
            ).fit(X, y)

    def test_invalid_task(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="Unknown task"):
            GeometricSubspaceSelector(
                n_components=1, task="nonexistent"
            ).fit(X, y)

    def test_invalid_solver(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="Unknown solver"):
            GeometricSubspaceSelector(
                n_components=1, solver="magic"
            ).fit(X, y)


# ── Transform before fit ──────────────────────────────────────────


class TestTransformBeforeFit:
    def test_transform_unfitted(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1)
        with pytest.raises(Exception):
            sel.transform(X)

    def test_transform_wrong_n_features(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            sel.transform(X[:, :2])

    def test_transform_1d(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        with pytest.raises(ValueError, match="2-D"):
            sel.transform(X[0])


# ── Constant / duplicate features ─────────────────────────────────


class TestDegenerateFeatures:
    def test_constant_column_warns(self, iris):
        X, y = iris
        X_const = np.hstack([X, np.ones((len(X), 1)) * 42])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeometricSubspaceSelector(n_components=1).fit(X_const, y)
            warning_msgs = [str(ww.message) for ww in w]
            assert any("near-zero variance" in m for m in warning_msgs)

    def test_all_constant_features(self):
        n = 30
        X = np.ones((n, 3))
        y = np.repeat([0, 1, 2], 10)
        # This should fail because after standardisation everything is 0/NaN
        with pytest.raises((ValueError, Exception)):
            GeometricSubspaceSelector(n_components=1).fit(X, y)


# ── Extreme regularisation ────────────────────────────────────────


class TestExtremeRegularisation:
    def test_zero_floor_runs(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(
            n_components=1, reg_floor_frac=0.0
        ).fit(X, y)
        assert sel.components_.shape[1] == 1

    def test_very_high_floor_warns(self):
        # Use a higher-dimensional case so >80% of eigvals fall below floor
        rng = np.random.RandomState(42)
        n, p, k = 60, 20, 3
        X = rng.randn(n, p)
        y = np.repeat([0, 1, 2], 20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GeometricSubspaceSelector(
                n_components=1, reg_floor_frac=0.99
            ).fit(X, y)
            warning_msgs = [str(ww.message) for ww in w]
            assert any("regularisation" in m.lower() for m in warning_msgs)


# ── Summary output ────────────────────────────────────────────────


# ── max_reduced_rank validation ────────────────────────────────────


class TestMaxReducedRank:
    def test_float_raises(self, iris):
        X, y = iris
        with pytest.raises(TypeError, match="integer"):
            GeometricSubspaceSelector(
                n_components=1, max_reduced_rank=2.5
            ).fit(X, y)

    def test_zero_raises(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="at least 1"):
            GeometricSubspaceSelector(
                n_components=1, max_reduced_rank=0
            ).fit(X, y)

    def test_negative_raises(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="at least 1"):
            GeometricSubspaceSelector(
                n_components=1, max_reduced_rank=-3
            ).fit(X, y)

    def test_valid_value(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(
            n_components=1, max_reduced_rank=3
        ).fit(X, y)
        assert sel.components_.shape[1] == 1


# ── Transform NaN/Inf ──────────────────────────────────────────────


class TestTransformNonFinite:
    def test_nan_in_transform(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            sel.transform(X_bad)

    def test_inf_in_transform(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        X_bad = X.copy()
        X_bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            sel.transform(X_bad)


# ── n_features_in_ ────────────────────────────────────────────────


class TestNFeaturesIn:
    def test_attribute_set(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        assert sel.n_features_in_ == X.shape[1]


# ── Zero scatter ──────────────────────────────────────────────────


class TestZeroScatter:
    def test_zero_scatter_with_zero_floor(self):
        """Identical within-class data + reg_floor_frac=0 should fail cleanly."""
        # Two classes, each with identical samples
        X = np.array([
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
            [3.0, 4.0], [3.0, 4.0], [3.0, 4.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="zero"):
            GeometricSubspaceSelector(
                n_components=1, reg_floor_frac=0.0
            ).fit(X, y)


# ── Baseline shape validation ─────────────────────────────────────


class TestBaselineValidation:
    def test_wrong_feature_dim(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        B_bad = np.eye(3, 2)  # 3 features, should be 4
        with pytest.raises(ValueError, match="rows"):
            ObserverReport.from_selector(sel, baselines={"bad": B_bad})

    def test_wrong_n_components(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        B_bad = np.eye(4, 1)  # 1 component, should be 2
        with pytest.raises(ValueError, match="columns"):
            ObserverReport.from_selector(sel, baselines={"bad": B_bad})

    def test_1d_baseline(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        with pytest.raises(ValueError, match="2-D"):
            ObserverReport.from_selector(
                sel, baselines={"bad": np.ones(4)}
            )


# ── Solver fallback warning ───────────────────────────────────────


class TestSolverFallbackWarning:
    """The fallback warning is only emitted when commuting_exact
    actually fails.  With a single aggregate matrix it should not
    fail on normal data, so we test the method_ attribute instead
    and verify the fallback path exists."""

    def test_eigenvector_fallback_sets_method(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(
            n_components=1, solver="eigenvector_fallback"
        ).fit(X, y)
        assert sel.method_ == "eigenvector_fallback"


# ── Caveat in all summaries ───────────────────────────────────────


class TestCaveatEverywhere:
    def test_audit_plain_caveat(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=1, floor_fracs=[0.01, 0.1],
        )
        s = audit.summary()
        assert "declared" in s.lower() or "valid" in s.lower()

    def test_ladder_plain_caveat(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        s = ladder.summary()
        assert "declared" in s.lower() or "valid" in s.lower()


# ── Summary output ────────────────────────────────────────────────


class TestSummaryContent:
    def test_report_plain_contains_caveat(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        report = ObserverReport.from_selector(sel)
        s = report.summary()
        assert "declared" in s.lower()
        assert "cross-validation" in s.lower() or "permutation" in s.lower()

    def test_report_technical_contains_caveat(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        report = ObserverReport.from_selector(sel)
        s = report.summary(technical=True)
        assert "declared" in s.lower()

    def test_report_plain_has_percentages(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        report = ObserverReport.from_selector(sel)
        s = report.summary()
        assert "%" in s

    def test_ladder_plain_explains_terms(self, iris):
        X, y = iris
        ladder = DimensionCostLadder.build(X, y, max_rank=2)
        s = ladder.summary()
        assert "retained" in s.lower()
        assert "cost range" in s.lower() or "penalty" in s.lower()

    def test_audit_plain_says_stable_or_unstable(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=1, floor_fracs=[0.01, 0.1],
        )
        s = audit.summary()
        assert "STABLE" in s or "UNSTABLE" in s
