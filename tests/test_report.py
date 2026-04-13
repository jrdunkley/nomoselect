"""Tests for nomoselect.report — ObserverReport."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from nomoselect import GeometricSubspaceSelector, ObserverReport


@pytest.fixture
def iris_fitted():
    d = load_iris()
    X, y = d.data, d.target
    sel = GeometricSubspaceSelector(n_components=2, task="fisher").fit(X, y)
    return sel, X, y


class TestFromSelector:
    def test_basic_report(self, iris_fitted):
        sel, X, y = iris_fitted
        report = ObserverReport.from_selector(sel)
        assert report.n_components == 2
        assert report.visible_score > 0
        assert 0 <= report.eta <= 1.0 + 1e-10
        assert report.reg_floor > 0

    def test_per_class_diagnostics(self, iris_fitted):
        sel, X, y = iris_fitted
        report = ObserverReport.from_selector(sel)
        assert len(report.per_class) == 3
        for pc in report.per_class:
            assert pc.visible_score >= 0
            assert pc.leakage >= 0

    def test_summary_string(self, iris_fitted):
        sel, X, y = iris_fitted
        report = ObserverReport.from_selector(sel)
        s = report.summary()
        assert "nomoselect report" in s
        assert "retained" in s.lower()
        # Technical mode
        s_tech = report.summary(technical=True)
        assert "ObserverReport" in s_tech
        assert "visible_score" in s_tech


class TestBaselineComparison:
    def test_pca_baseline(self, iris_fitted):
        sel, X, y = iris_fitted
        from sklearn.preprocessing import StandardScaler
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2).fit(X_std)
        B_pca = pca.components_.T  # (p, 2)

        report = ObserverReport.from_selector(sel, baselines={"PCA": B_pca})
        assert len(report.baselines) == 1
        bc = report.baselines[0]
        assert bc.name == "PCA"
        assert isinstance(bc.geo_dominates, bool)
        assert isinstance(bc.baseline_dominates, bool)

    def test_summary_with_baseline(self, iris_fitted):
        sel, X, y = iris_fitted
        from sklearn.preprocessing import StandardScaler
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2).fit(X_std)
        B_pca = pca.components_.T

        report = ObserverReport.from_selector(sel, baselines={"PCA": B_pca})
        s = report.summary()
        assert "PCA" in s
        assert "more" in s or "dominates" in s
        # Technical mode still shows dominance
        s_tech = report.summary(technical=True)
        assert "dominates" in s_tech
