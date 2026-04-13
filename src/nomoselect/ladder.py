"""
DimensionCostLadder — is another dimension worth it?

Fits the selector at ranks 1, 2, ..., max_rank, then uses nomogeo to
compute exact cost intervals: for each candidate rank, the range of
trade-off parameters where that rank is optimal.

This is the task-aware replacement for scree plots and information
criteria (AIC, BIC).  Instead of a heuristic elbow or a point estimate,
you get an exact phase diagram showing when each rank wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import nomogeo

from .selector import GeometricSubspaceSelector


@dataclass
class LadderRung:
    """One candidate rank in the ladder."""

    rank: int
    visible_score: float
    leakage: float
    eta: float
    interval_lower: float
    interval_upper: float
    is_viable: bool


@dataclass
class DimensionCostLadder:
    """Task-aware rank selection diagram.

    Attributes
    ----------
    rungs : list of LadderRung
        One entry per candidate rank.
    winner_at_zero : list of int
        Rank(s) that win when cost is zero (pure structure maximisation).
    recommended_rank : int
        The viable rank with the widest cost interval — a simple default.
    """

    rungs: list[LadderRung] = field(default_factory=list)
    winner_at_zero: list[int] = field(default_factory=list)
    recommended_rank: int = 1

    @classmethod
    def build(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        max_rank: int | None = None,
        task: str = "fisher",
        reg_floor_frac: float = 0.01,
        **selector_kwargs,
    ) -> "DimensionCostLadder":
        """Build the ladder by fitting at each candidate rank.

        Parameters
        ----------
        X, y : training data and labels.
        max_rank : maximum rank to evaluate.
            Default: min(k-1, 10) where k is the number of classes.
        task : task family name.
        reg_floor_frac : regularisation floor.
        **selector_kwargs : extra arguments for GeometricSubspaceSelector.

        Returns
        -------
        DimensionCostLadder
        """
        k = len(np.unique(y))
        if max_rank is None:
            max_rank = min(k - 1, 10)
        ranks = list(range(1, max_rank + 1))

        scores_list = []
        dims_list = []
        selectors = {}

        for rank in ranks:
            sel = GeometricSubspaceSelector(
                n_components=rank,
                task=task,
                reg_floor_frac=reg_floor_frac,
                **selector_kwargs,
            ).fit(X, y)
            selectors[rank] = sel
            scores_list.append(sel.closure_scores_.visible_score)
            dims_list.append(float(rank))

        scores_arr = np.array(scores_list)
        dims_arr = np.array(dims_list)

        ladder_result = nomogeo.declared_ladder_dimension_cost_intervals(
            scores_arr, dims_arr
        )

        ladder = cls()
        for i, rank in enumerate(ranks):
            sc = selectors[rank].closure_scores_
            rung = LadderRung(
                rank=rank,
                visible_score=sc.visible_score,
                leakage=sc.leakage,
                eta=sc.eta,
                interval_lower=float(ladder_result.interval_lower[i]),
                interval_upper=float(ladder_result.interval_upper[i]),
                is_viable=bool(ladder_result.interval_nonempty[i]),
            )
            ladder.rungs.append(rung)

        ladder.winner_at_zero = [
            ranks[i] for i in ladder_result.winner_at_zero
        ]

        # Heuristic: widest viable interval
        viable = [r for r in ladder.rungs if r.is_viable]
        if viable:
            def interval_width(r: LadderRung) -> float:
                if np.isinf(r.interval_upper):
                    return 1e12
                return r.interval_upper - r.interval_lower
            ladder.recommended_rank = max(viable, key=interval_width).rank
        elif ladder.rungs:
            ladder.recommended_rank = ladder.rungs[0].rank

        return ladder

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
        lines = [
            f"Dimension cost ladder",
            f"  Best rank if cost is zero: {self.winner_at_zero}",
            f"  Recommended rank: {self.recommended_rank}",
            "",
            f"  {'dims':>4s}  {'retained':>10s}  {'lost':>10s}  {'cost range':>20s}  {'viable':>6s}",
        ]
        for r in self.rungs:
            pct = 100.0 * r.visible_score / max(r.visible_score + r.leakage, 1e-300)
            pct_lost = 100.0 - pct
            hi = f"{r.interval_upper:.4f}" if not np.isinf(r.interval_upper) else "no limit"
            cost_range = f"[{r.interval_lower:.4f}, {hi}]"
            lines.append(
                f"  {r.rank:4d}  {pct:9.1f}%  {pct_lost:9.1f}%  {cost_range:>20s}  "
                f"{'yes' if r.is_viable else 'no':>6s}"
            )
        lines.append("")
        lines.append(
            "  'Retained' = fraction of declared target structure kept."
        )
        lines.append(
            "  'Cost range' = where this rank is optimal as you increase"
        )
        lines.append(
            "  the penalty for extra dimensions.  A wider range means the"
        )
        lines.append(
            "  rank is robust to that trade-off."
        )
        lines.append("")
        lines.append(
            "  Note: these intervals are exact for the declared task."
        )
        lines.append(
            "  They do not confirm the task itself is valid."
        )
        return "\n".join(lines)

    def _summary_technical(self) -> str:
        lines = [
            f"DeclaredLadderDimensionCost  "
            f"(winner@c=0: {self.winner_at_zero}, "
            f"recommended: {self.recommended_rank})",
            f"  {'rank':>4s}  {'vis_score':>10s}  {'eta':>8s}  "
            f"{'c_low':>10s}  {'c_high':>10s}  {'viable':>6s}",
        ]
        for r in self.rungs:
            hi = (f"{r.interval_upper:.4f}"
                  if not np.isinf(r.interval_upper) else "inf")
            lines.append(
                f"  {r.rank:4d}  {r.visible_score:10.4f}  {r.eta:8.4f}  "
                f"{r.interval_lower:10.4f}  {hi:>10s}  "
                f"{'yes' if r.is_viable else 'no':>6s}"
            )
        return "\n".join(lines)
