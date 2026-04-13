"""
ObserverReport — diagnostic summary for a fitted GeometricSubspaceSelector.

Tells you exactly what was kept and what was lost in the reduction,
with optional comparison against PCA or other baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import nomogeo

from .selector import GeometricSubspaceSelector
from .tasks import TaskFamily


@dataclass
class BaselineComparison:
    """Comparison of the fitted view against one baseline."""

    name: str
    scores: nomogeo.ClosureScoresResult
    dominance: nomogeo.ObserverComparisonResult
    geo_dominates: bool
    baseline_dominates: bool


@dataclass
class PerClassDiagnostic:
    """Per-class structure retention under the fitted view."""

    label: str
    weight: float
    visible_score: float
    leakage: float
    eta: float


@dataclass
class ObserverReport:
    """Diagnostic report for a fitted GeometricSubspaceSelector.

    Attributes
    ----------
    n_components : int
        Number of dimensions kept.
    method : str
        Solver used.
    visible_score : float
        How much of the target structure is kept.
    leakage : float
        How much of the target structure is lost.
    eta : float
        Fraction lost:  leakage / (leakage + visible_score).
    total_curvature : float
        Total target structure in the full space.
    reg_floor : float
        Regularisation floor applied.
    n_above_floor : int
        Scatter eigenvalues above the floor.
    per_class : list of PerClassDiagnostic
        Per-class breakdown.
    baselines : list of BaselineComparison
        Comparisons against PCA or other views.
    """

    n_components: int
    method: str
    visible_score: float
    leakage: float
    eta: float
    total_curvature: float
    reg_floor: float
    n_above_floor: int
    eigvals_sw_top5: list[float]
    per_class: list[PerClassDiagnostic] = field(default_factory=list)
    baselines: list[BaselineComparison] = field(default_factory=list)

    @classmethod
    def from_selector(
        cls,
        sel: GeometricSubspaceSelector,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        baselines: dict[str, np.ndarray] | None = None,
    ) -> "ObserverReport":
        """Build a report from a fitted selector.

        Parameters
        ----------
        sel : GeometricSubspaceSelector
            Must be already fitted.
        X, y : optional
            Not currently needed; reserved for future extensions.
        baselines : dict mapping name -> (p, n_components) array, optional
            Column-orthonormal projection matrices in standardised feature
            space, e.g. ``{"PCA": pca.components_.T}``.
        """
        sc = sel.closure_scores_
        H = sel._H
        family_agg = sel.task_family_.aggregate

        # Per-class diagnostics
        per_class = []
        if sel.task_family_.individual:
            B_w = sel._B_whitened
            for i, (T_i, label) in enumerate(
                zip(sel.task_family_.individual, sel.task_family_.labels)
            ):
                sc_i = nomogeo.closure_scores(H, [T_i], B_w)
                per_class.append(PerClassDiagnostic(
                    label=label,
                    weight=float(sel.task_family_.weights[i]),
                    visible_score=sc_i.visible_score,
                    leakage=sc_i.leakage,
                    eta=sc_i.eta,
                ))

        top5 = sel.eigvals_sw_[:5].tolist()

        report = cls(
            n_components=sel.n_components,
            method=sel.method_,
            visible_score=sc.visible_score,
            leakage=sc.leakage,
            eta=sc.eta,
            total_curvature=sc.total_curvature,
            reg_floor=sel.reg_floor_,
            n_above_floor=sel.n_above_floor_,
            eigvals_sw_top5=top5,
            per_class=per_class,
        )

        # Baseline comparisons
        if baselines:
            n_features = sel.n_features_in_
            n_comp = sel.n_components
            for name, B_base_full in baselines.items():
                B_base_full = np.asarray(B_base_full, dtype=float)
                if B_base_full.ndim != 2:
                    raise ValueError(
                        f"Baseline '{name}' must be a 2-D array, "
                        f"got shape {B_base_full.shape}."
                    )
                if B_base_full.shape[0] != n_features:
                    raise ValueError(
                        f"Baseline '{name}' has {B_base_full.shape[0]} "
                        f"rows, but the selector was fitted on "
                        f"{n_features} features.  The baseline must "
                        f"be a (n_features, n_components) matrix in "
                        f"standardised feature space."
                    )
                if B_base_full.shape[1] != n_comp:
                    raise ValueError(
                        f"Baseline '{name}' has {B_base_full.shape[1]} "
                        f"columns, but the selector uses "
                        f"n_components={n_comp}.  The baseline must "
                        f"have the same number of components for a "
                        f"fair comparison."
                    )
                B_base_red = sel._V_reduce.T @ B_base_full
                B_base_w = sel._S_W_sqrt_diag[:, None] * B_base_red
                B_base_w, _ = np.linalg.qr(B_base_w)
                sc_base = nomogeo.closure_scores(H, family_agg, B_base_w)
                comp = nomogeo.compare_observers(
                    H, family_agg, B_base_w, sel._B_whitened
                )
                report.baselines.append(BaselineComparison(
                    name=name,
                    scores=sc_base,
                    dominance=comp,
                    geo_dominates=comp.right_dominates,
                    baseline_dominates=comp.left_dominates,
                ))

        return report

    def summary(self, technical: bool = False) -> str:
        """Human-readable summary.

        Parameters
        ----------
        technical : bool, default False
            If True, use internal mathematical names.
            If False (default), use plain-language descriptions.
        """
        if technical:
            return self._summary_technical()
        return self._summary_plain()

    def _summary_plain(self) -> str:
        """Plain-language summary suitable for non-specialist users."""
        pct_kept = self._pct_kept()
        pct_lost = 100.0 - pct_kept

        lines = [
            f"nomoselect report  ({self.n_components} dimension{'s' if self.n_components > 1 else ''} kept)",
            "",
            f"  Target structure retained : {pct_kept:5.1f}%",
            f"  Target structure lost     : {pct_lost:5.1f}%",
        ]

        if self.total_curvature > 0:
            lines.append(
                f"  Total target strength     : {self.total_curvature:.4f}"
            )

        lines.append(
            f"  Regularisation floor      : {self.reg_floor:.2e}  "
            f"({self.n_above_floor} eigenvalues above)"
        )

        if self.per_class:
            lines.append("")
            lines.append("  Per-class retention:")
            for pc in self.per_class:
                pc_pct = self._safe_pct(pc.visible_score, pc.visible_score + pc.leakage)
                lines.append(
                    f"    {pc.label:20s}  retained {pc_pct:5.1f}%"
                    f"  (weight={pc.weight:.3f})"
                )

        if self.baselines:
            lines.append("")
            lines.append("  Comparisons:")
            for bc in self.baselines:
                bc_pct = self._safe_pct(
                    bc.scores.visible_score,
                    bc.scores.visible_score + bc.scores.leakage,
                )
                if bc.geo_dominates:
                    verdict = "nomoselect keeps strictly more"
                elif bc.baseline_dominates:
                    verdict = f"{bc.name} keeps strictly more"
                else:
                    verdict = "neither strictly dominates"
                lines.append(
                    f"    vs {bc.name:12s}  "
                    f"{bc.name} retains {bc_pct:.1f}%  [{verdict}]"
                )

        lines.append("")
        lines.append("  Note: these diagnostics are exact for the task you")
        lines.append("  declared.  They do not prove the task itself is valid.")
        lines.append("  Cross-validation or permutation tests are still needed")
        lines.append("  to confirm that the structure is real.")
        return "\n".join(lines)

    def _summary_technical(self) -> str:
        """Technical summary using mathematical names."""
        lines = [
            f"ObserverReport  (rank={self.n_components}, solver={self.method})",
            f"  visible_score   = {self.visible_score:.6f}",
            f"  leakage         = {self.leakage:.6f}",
            f"  eta (hidden)    = {self.eta:.6f}",
            f"  total_curvature = {self.total_curvature:.6f}",
            f"  reg_floor       = {self.reg_floor:.2e}  "
            f"({self.n_above_floor} eigvals above)",
        ]
        if self.per_class:
            lines.append("  Per-class:")
            for pc in self.per_class:
                lines.append(
                    f"    {pc.label:20s}  w={pc.weight:.3f}  "
                    f"vis={pc.visible_score:.4f}  "
                    f"leak={pc.leakage:.4f}  eta={pc.eta:.4f}"
                )
        if self.baselines:
            lines.append("  Baselines:")
            for bc in self.baselines:
                dom = ("GEO dominates" if bc.geo_dominates else
                       (f"{bc.name} dominates" if bc.baseline_dominates
                        else "neither dominates"))
                lines.append(
                    f"    {bc.name:12s}  vis={bc.scores.visible_score:.4f}  "
                    f"eta={bc.scores.eta:.4f}  [{dom}]"
                )
        lines.append("")
        lines.append("  Note: certificates are exact for the declared task.")
        lines.append("  They do not validate the task itself.")
        return "\n".join(lines)

    # ── Helpers ──

    def _pct_kept(self) -> float:
        if self.total_curvature <= 0:
            return 100.0
        return 100.0 * self.visible_score / self.total_curvature

    @staticmethod
    def _safe_pct(num: float, denom: float) -> float:
        if denom <= 0:
            return 100.0
        return 100.0 * num / denom
