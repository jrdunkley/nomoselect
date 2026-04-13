"""
Task family constructors for nomoselect.

A *task family* is a list of symmetric matrices [T_1, ..., T_k] that encode
what the observer should see.  In the pre-whitened convention used throughout
nomoselect the ambient precision is H = I, and each T_i is already expressed
in the whitened coordinate system:

    T_i  =  S_W^{-1/2}  M_i  S_W^{-1/2}

where M_i is a between-class outer product and S_W is the pooled within-class
covariance (regularised as needed).

Aggregate vs individual
-----------------------
closure_adapted_observer(H, family, rank, mode="commuting_exact") requires
the family members to mutually commute.  A single aggregate matrix always
satisfies this, so every constructor returns *both*:

    aggregate   – list with one element (the weighted sum)
    individual  – list with one element per class or pair

The caller chooses which to pass to nomogeo.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class TaskFamily(NamedTuple):
    """Container for a task family with aggregate and individual forms."""

    aggregate: list[np.ndarray]
    """Single-element list: the weighted sum of all individual tasks."""

    individual: list[np.ndarray]
    """Per-class (or per-pair) tasks, one matrix each."""

    weights: np.ndarray
    """Weight assigned to each individual task."""

    labels: list[str]
    """Human-readable label for each individual task."""


def _validate_inputs(class_means_w: np.ndarray, class_sizes: np.ndarray):
    """Shared validation for task family constructors."""
    class_means_w = np.asarray(class_means_w, dtype=float)
    class_sizes = np.asarray(class_sizes, dtype=float)
    if class_means_w.ndim != 2:
        raise ValueError("class_means_w must be 2-D  (k × d)")
    k, d = class_means_w.shape
    if class_sizes.shape != (k,):
        raise ValueError(f"class_sizes must have length {k}")
    if np.any(class_sizes <= 0):
        raise ValueError("class_sizes must be strictly positive")
    return class_means_w, class_sizes, k, d


def _grand_mean_w(class_means_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted grand mean in whitened space."""
    return weights @ class_means_w


def _build_family(
    class_means_w: np.ndarray,
    grand_mean: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
) -> TaskFamily:
    """Construct TaskFamily from centred outer products."""
    k, d = class_means_w.shape
    individual = []
    for i in range(k):
        diff = class_means_w[i] - grand_mean
        individual.append(weights[i] * np.outer(diff, diff))
    agg = sum(individual)
    agg = 0.5 * (agg + agg.T)  # symmetrise
    return TaskFamily(
        aggregate=[agg],
        individual=individual,
        weights=weights,
        labels=labels,
    )


# ── Public constructors ─────────────────────────────────────────────


def fisher_task_family(
    class_means_w: np.ndarray,
    class_sizes: np.ndarray,
    class_labels: list[str] | None = None,
) -> TaskFamily:
    """Standard Fisher task family: weight_i = n_i / n.

    Parameters
    ----------
    class_means_w : (k, d) array
        Class centroids in the whitened coordinate system.
    class_sizes : (k,) array
        Number of samples per class.
    class_labels : list of str, optional
        Human-readable class names.  Defaults to "class_0", "class_1", ...

    Returns
    -------
    TaskFamily
        aggregate  = [S_W^{-1/2} S_B S_W^{-1/2}]  (Fisher criterion)
        individual = one rank-1 matrix per class, sample-weighted
    """
    class_means_w, class_sizes, k, d = _validate_inputs(class_means_w, class_sizes)
    n = class_sizes.sum()
    weights = class_sizes / n
    labels = class_labels or [f"class_{i}" for i in range(k)]
    grand = _grand_mean_w(class_means_w, weights)
    return _build_family(class_means_w, grand, weights, labels)


def equal_weight_task_family(
    class_means_w: np.ndarray,
    class_sizes: np.ndarray,
    class_labels: list[str] | None = None,
) -> TaskFamily:
    """Equal-weight task family: weight_i = 1/k.

    Gives every class equal importance regardless of sample size.
    May recover minority-class structure invisible under Fisher weighting.
    """
    class_means_w, class_sizes, k, d = _validate_inputs(class_means_w, class_sizes)
    weights = np.full(k, 1.0 / k)
    labels = class_labels or [f"class_{i}" for i in range(k)]
    grand = _grand_mean_w(class_means_w, weights)
    return _build_family(class_means_w, grand, weights, labels)


def minority_emphasis_family(
    class_means_w: np.ndarray,
    class_sizes: np.ndarray,
    class_labels: list[str] | None = None,
) -> TaskFamily:
    """Inverse-frequency task family: weight_i proportional to 1 / n_i.

    Emphasises rare classes.  The weights are normalised to sum to 1.
    """
    class_means_w, class_sizes, k, d = _validate_inputs(class_means_w, class_sizes)
    raw = 1.0 / class_sizes
    weights = raw / raw.sum()
    labels = class_labels or [f"class_{i}" for i in range(k)]
    grand = _grand_mean_w(class_means_w, weights)
    return _build_family(class_means_w, grand, weights, labels)


def pairwise_task_family(
    class_means_w: np.ndarray,
    class_sizes: np.ndarray,
    class_labels: list[str] | None = None,
) -> TaskFamily:
    """Pairwise task family: one task per class pair.

    Each task T_{ij} = (mu_i - mu_j)(mu_i - mu_j)^T, equally weighted.
    The aggregate is the sum over all k(k-1)/2 pairs, normalised.
    """
    class_means_w, class_sizes, k, d = _validate_inputs(class_means_w, class_sizes)
    labels_in = class_labels or [f"class_{i}" for i in range(k)]
    individual = []
    pair_labels = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = class_means_w[i] - class_means_w[j]
            individual.append(np.outer(diff, diff))
            pair_labels.append(f"{labels_in[i]}_vs_{labels_in[j]}")
    n_pairs = len(individual)
    weights = np.full(n_pairs, 1.0 / n_pairs)
    for idx in range(n_pairs):
        individual[idx] = individual[idx] * weights[idx]
    agg = sum(individual)
    agg = 0.5 * (agg + agg.T)
    return TaskFamily(
        aggregate=[agg],
        individual=individual,
        weights=weights,
        labels=pair_labels,
    )


def custom_task_family(
    tasks: list[np.ndarray],
    weights: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> TaskFamily:
    """User-supplied task family with optional weights.

    Parameters
    ----------
    tasks : list of (d, d) symmetric arrays
        Each matrix encodes one task the observer should retain.
    weights : (len(tasks),) array, optional
        Non-negative weights.  Default: uniform 1/len(tasks).
    labels : list of str, optional
        Human-readable label per task.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty")
    d = tasks[0].shape[0]
    for i, t in enumerate(tasks):
        t = np.asarray(t, dtype=float)
        if t.shape != (d, d):
            raise ValueError(f"Task {i} has shape {t.shape}, expected ({d}, {d})")
        tasks[i] = 0.5 * (t + t.T)
    k = len(tasks)
    if weights is None:
        weights = np.full(k, 1.0 / k)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (k,):
            raise ValueError(f"weights must have length {k}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
    labels = labels or [f"task_{i}" for i in range(k)]
    individual = [w * t for w, t in zip(weights, tasks)]
    agg = sum(individual)
    agg = 0.5 * (agg + agg.T)
    return TaskFamily(
        aggregate=[agg],
        individual=individual,
        weights=weights,
        labels=labels,
    )
