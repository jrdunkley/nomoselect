"""Tests for nomoselect.selector — GeometricSubspaceSelector."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nomoselect import GeometricSubspaceSelector


@pytest.fixture
def iris():
    d = load_iris()
    return d.data, d.target


@pytest.fixture
def wine():
    d = load_wine()
    return d.data, d.target


@pytest.fixture
def breast_cancer():
    d = load_breast_cancer()
    return d.data, d.target


class TestBasicFit:
    def test_fit_returns_self(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2)
        result = sel.fit(X, y)
        assert result is sel

    def test_components_shape(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        assert sel.components_.shape == (X.shape[1], 2)

    def test_components_orthonormal(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        gram = sel.components_.T @ sel.components_
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)

    def test_transform_shape(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        Xp = sel.transform(X)
        assert Xp.shape == (X.shape[0], 2)

    def test_fit_transform_equals_fit_then_transform(self, iris):
        X, y = iris
        sel1 = GeometricSubspaceSelector(n_components=2, reg_floor_frac=0.01)
        Xp1 = sel1.fit_transform(X, y)
        sel2 = GeometricSubspaceSelector(n_components=2, reg_floor_frac=0.01)
        sel2.fit(X, y)
        Xp2 = sel2.transform(X)
        np.testing.assert_allclose(Xp1, Xp2, atol=1e-10)


class TestClosureDiagnostics:
    def test_eta_non_negative(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        assert sel.closure_scores_.eta >= 0.0

    def test_eta_leq_one(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=1).fit(X, y)
        assert sel.closure_scores_.eta <= 1.0 + 1e-10

    def test_visible_plus_leakage_eq_total(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        sc = sel.closure_scores_
        np.testing.assert_allclose(
            sc.visible_score + sc.leakage, sc.total_curvature, rtol=1e-10
        )


class TestTaskFamilies:
    def test_fisher(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, task="fisher").fit(X, y)
        assert sel.task_family_.labels == ["0", "1", "2"]

    def test_equal_weight(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, task="equal_weight").fit(X, y)
        np.testing.assert_allclose(sel.task_family_.weights, [1/3, 1/3, 1/3])

    def test_minority(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, task="minority").fit(X, y)
        assert sel.task_family_ is not None

    def test_pairwise(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, task="pairwise").fit(X, y)
        # 3 classes → 3 pairs
        assert len(sel.task_family_.individual) == 3

    def test_invalid_task_raises(self, iris):
        X, y = iris
        with pytest.raises(ValueError, match="Unknown task"):
            GeometricSubspaceSelector(n_components=2, task="bogus").fit(X, y)


class TestSolverFallback:
    def test_eigenvector_fallback_explicit(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(
            n_components=2, solver="eigenvector_fallback"
        ).fit(X, y)
        assert sel.method_ == "eigenvector_fallback"

    def test_commuting_exact_method(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(
            n_components=2, solver="commuting_exact"
        ).fit(X, y)
        # Should succeed with aggregate (single matrix) family
        assert sel.method_ in ("commuting_exact", "eigenvector_fallback")


class TestFisherAlignment:
    """GEO with Fisher task should closely align with sklearn LDA."""

    def test_iris_rank2_alignment(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, task="fisher").fit(X, y)
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
        B_lda = lda.scalings_[:, :2]
        B_lda, _ = np.linalg.qr(B_lda)
        # Principal angles
        svs = np.linalg.svd(sel.components_.T @ B_lda, compute_uv=False)
        svs = np.clip(svs, 0, 1)
        cosine = float(np.prod(svs))
        # Should be very close — any gap is regularisation, not criterion
        assert cosine > 0.8, f"GEO-LDA cosine={cosine:.4f}, expected > 0.8"

    def test_wine_rank1_high_accuracy(self, wine):
        X, y = wine
        sel = GeometricSubspaceSelector(n_components=1, task="fisher").fit(X, y)
        Xp = sel.transform(X)
        # Simple nearest-centroid accuracy check
        centroids = {}
        for c in np.unique(y):
            centroids[c] = Xp[y == c].mean()
        preds = np.array([
            min(centroids, key=lambda c: abs(Xp[i, 0] - centroids[c]))
            for i in range(len(y))
        ])
        acc = (preds == y).mean()
        assert acc > 0.85, f"Wine rank-1 accuracy={acc:.4f}, expected > 0.85"


class TestHighDimensional:
    """Test the dual trick with p >> n synthetic data."""

    def test_synthetic_p_gt_n(self):
        rng = np.random.RandomState(42)
        n, p, k = 30, 200, 3
        X = rng.randn(n, p)
        y = np.repeat([0, 1, 2], 10)
        # Inject signal in first 2 features
        for c in range(k):
            X[y == c, c] += 3.0

        sel = GeometricSubspaceSelector(n_components=2).fit(X, y)
        assert sel.components_.shape == (p, 2)
        assert sel.closure_scores_.eta < 1.0


class TestRegularisation:
    def test_attributes_present(self, iris):
        X, y = iris
        sel = GeometricSubspaceSelector(n_components=2, reg_floor_frac=0.05).fit(X, y)
        assert sel.reg_floor_ > 0
        assert sel.n_above_floor_ > 0
        assert len(sel.eigvals_sw_) > 0

    def test_different_floors_same_shape(self, iris):
        X, y = iris
        sel1 = GeometricSubspaceSelector(n_components=2, reg_floor_frac=0.001).fit(X, y)
        sel2 = GeometricSubspaceSelector(n_components=2, reg_floor_frac=0.1).fit(X, y)
        assert sel1.components_.shape == sel2.components_.shape
