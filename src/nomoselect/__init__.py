"""
nomoselect — Task-aware subspace selection with exact diagnostics.

Built on the nomogeo observer-geometry kernel.

Public API:
    GeometricSubspaceSelector  — fit/transform interface for subspace selection
    ObserverReport             — diagnostic report for a fitted observer
    RegularisationAudit        — regularisation-path stability analysis
    DimensionCostLadder        — task-aware rank phase diagram

Task families:
    fisher_task_family         — standard Fisher (sample-weighted S_B)
    equal_weight_task_family   — equal weight per class (1/k)
    minority_emphasis_family   — inverse-frequency weighting
    pairwise_task_family       — one task per class pair
"""

from .selector import GeometricSubspaceSelector
from .report import ObserverReport
from .audit import RegularisationAudit
from .ladder import DimensionCostLadder
from .tasks import (
    fisher_task_family,
    equal_weight_task_family,
    minority_emphasis_family,
    pairwise_task_family,
)

__version__ = "0.1.0"

__all__ = [
    "GeometricSubspaceSelector",
    "ObserverReport",
    "RegularisationAudit",
    "DimensionCostLadder",
    "fisher_task_family",
    "equal_weight_task_family",
    "minority_emphasis_family",
    "pairwise_task_family",
    "__version__",
]
