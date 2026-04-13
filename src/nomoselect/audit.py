"""
RegularisationAudit — stability analysis across the regularisation path.

Sweeps the Tikhonov floor parameter and records:
  - closure scores (visible, leakage, eta)
  - effective rank (eigenvalues above floor)
  - subspace alignment (cosine similarity to a reference fit)
  - optional cross-validated accuracy

This answers the question: "Is my result stable, or is it an artefact of
the regularisation choice?"
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .selector import GeometricSubspaceSelector


@dataclass
class AuditPoint:
    """Diagnostics at one regularisation setting."""

    reg_floor_frac: float
    reg_floor_abs: float
    n_above_floor: int
    visible_score: float
    leakage: float
    eta: float
    subspace_cosine: float  # alignment with reference fit
    components: np.ndarray  # (p, n_components) in standardised space


@dataclass
class RegularisationAudit:
    """Full regularisation-path audit.

    Attributes
    ----------
    points : list of AuditPoint
        One entry per regularisation setting, ordered by reg_floor_frac.
    reference_floor_frac : float
        The floor fraction used for the reference fit.
    stable : bool
        True if the subspace cosine stays above ``stability_threshold``
        across the sweep.
    stability_threshold : float
        Threshold used for the ``stable`` flag.
    """

    points: list[AuditPoint] = field(default_factory=list)
    reference_floor_frac: float = 0.01
    stable: bool = True
    stability_threshold: float = 0.9

    @classmethod
    def run(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int = 1,
        task: str = "fisher",
        floor_fracs: list[float] | None = None,
        reference_floor_frac: float = 0.01,
        stability_threshold: float = 0.9,
        **selector_kwargs,
    ) -> "RegularisationAudit":
        """Run a regularisation sweep.

        Parameters
        ----------
        X, y : training data and labels.
        n_components : subspace rank to test.
        task : task family name.
        floor_fracs : list of Tikhonov floor fractions to sweep.
            Default: [1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].
        reference_floor_frac : the floor used as the alignment reference.
        stability_threshold : minimum cosine for the ``stable`` flag.
        **selector_kwargs : extra arguments passed to GeometricSubspaceSelector.

        Returns
        -------
        RegularisationAudit
        """
        if floor_fracs is None:
            floor_fracs = [1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        # Fit reference
        ref_sel = GeometricSubspaceSelector(
            n_components=n_components,
            task=task,
            reg_floor_frac=reference_floor_frac,
            **selector_kwargs,
        ).fit(X, y)
        B_ref = ref_sel.components_  # (p, n_components)

        audit = cls(
            reference_floor_frac=reference_floor_frac,
            stability_threshold=stability_threshold,
        )

        for ff in sorted(floor_fracs):
            sel = GeometricSubspaceSelector(
                n_components=n_components,
                task=task,
                reg_floor_frac=ff,
                **selector_kwargs,
            ).fit(X, y)

            cosine = _subspace_cosine(B_ref, sel.components_)

            audit.points.append(AuditPoint(
                reg_floor_frac=ff,
                reg_floor_abs=sel.reg_floor_,
                n_above_floor=sel.n_above_floor_,
                visible_score=sel.closure_scores_.visible_score,
                leakage=sel.closure_scores_.leakage,
                eta=sel.closure_scores_.eta,
                subspace_cosine=cosine,
                components=sel.components_,
            ))

        # Determine stability
        cosines = [p.subspace_cosine for p in audit.points]
        audit.stable = all(c >= stability_threshold for c in cosines)

        return audit

    def summary(self, technical: bool = False) -> str:
        """Human-readable summary.

        Parameters
        ----------
        technical : bool, default False
            If True, use mathematical names.  If False, plain language.
        """
        if technical:
            return self._summary_technical()
        return self._summary_plain()

    def _summary_plain(self) -> str:
        stable_word = "STABLE" if self.stable else "UNSTABLE"
        lines = [
            f"Regularisation stability check: {stable_word}",
            f"  Reference floor: {self.reference_floor_frac}  "
            f"(threshold: alignment > {self.stability_threshold})",
            "",
            f"  {'floor':>8s}  {'n_eff':>5s}  {'retained':>10s}  {'alignment':>10s}",
        ]
        for p in self.points:
            marker = " (ref)" if p.reg_floor_frac == self.reference_floor_frac else ""
            pct = 100.0 * p.visible_score / max(
                p.visible_score + p.leakage, 1e-300
            )
            lines.append(
                f"  {p.reg_floor_frac:8.4f}  {p.n_above_floor:5d}  "
                f"{pct:9.1f}%  {p.subspace_cosine:10.4f}{marker}"
            )
        lines.append("")
        lines.append(
            "  'Retained' = % of target structure kept at that floor."
        )
        lines.append(
            "  'Alignment' = subspace cosine vs the reference fit."
        )
        lines.append(
            "  Values near 1.0 mean the result barely changes."
        )
        lines.append("")
        lines.append(
            "  Note: stability confirms the result is not an artefact of"
        )
        lines.append(
            "  regularisation.  It does not confirm the declared task is"
        )
        lines.append(
            "  valid.  Use cross-validation or permutation tests for that."
        )
        return "\n".join(lines)

    def _summary_technical(self) -> str:
        lines = [
            f"RegularisationAudit  (ref_floor={self.reference_floor_frac}, "
            f"stable={self.stable}, threshold={self.stability_threshold})",
            f"  {'floor':>8s}  {'n_eff':>5s}  {'vis':>8s}  {'eta':>8s}  {'cos':>8s}",
        ]
        for p in self.points:
            marker = " *" if p.reg_floor_frac == self.reference_floor_frac else ""
            lines.append(
                f"  {p.reg_floor_frac:8.4f}  {p.n_above_floor:5d}  "
                f"{p.visible_score:8.4f}  {p.eta:8.4f}  {p.subspace_cosine:8.4f}{marker}"
            )
        return "\n".join(lines)


def _subspace_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Principal-angle cosine between two subspaces of equal rank.

    Returns the product of cosines of all principal angles — a single
    scalar in [0, 1] that is 1.0 iff the subspaces are identical.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # QR to ensure orthonormal
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    svs = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    # Clamp to [0, 1] for numerical safety
    svs = np.clip(svs, 0.0, 1.0)
    return float(np.prod(svs))
