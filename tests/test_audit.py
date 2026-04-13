"""Tests for nomoselect.audit — RegularisationAudit."""
import numpy as np
import pytest
from sklearn.datasets import load_iris

from nomoselect import RegularisationAudit


@pytest.fixture
def iris():
    d = load_iris()
    return d.data, d.target


class TestAuditRun:
    def test_basic_audit(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=2,
            floor_fracs=[0.001, 0.01, 0.1],
        )
        assert len(audit.points) == 3

    def test_points_ordered(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=1,
            floor_fracs=[0.1, 0.001, 0.01],
        )
        fracs = [p.reg_floor_frac for p in audit.points]
        assert fracs == sorted(fracs)

    def test_cosine_at_reference_is_one(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=1,
            floor_fracs=[0.01],
            reference_floor_frac=0.01,
        )
        # At the reference setting, cosine should be 1.0
        np.testing.assert_allclose(audit.points[0].subspace_cosine, 1.0, atol=1e-8)

    def test_stability_flag(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=2,
            floor_fracs=[0.001, 0.01, 0.1],
            stability_threshold=0.5,
        )
        assert isinstance(audit.stable, bool)

    def test_summary_string(self, iris):
        X, y = iris
        audit = RegularisationAudit.run(
            X, y, n_components=1,
            floor_fracs=[0.01, 0.1],
        )
        s = audit.summary()
        assert "STABLE" in s or "UNSTABLE" in s
        # Technical mode still works
        s_tech = audit.summary(technical=True)
        assert "RegularisationAudit" in s_tech


class TestSubspaceCosine:
    def test_identical_subspaces(self):
        from nomoselect.audit import _subspace_cosine
        A = np.eye(5, 2)
        assert abs(_subspace_cosine(A, A) - 1.0) < 1e-10

    def test_orthogonal_subspaces(self):
        from nomoselect.audit import _subspace_cosine
        A = np.eye(4)[:, :2]   # columns 0,1
        B = np.eye(4)[:, 2:]   # columns 2,3
        cos = _subspace_cosine(A, B)
        assert cos < 1e-10
