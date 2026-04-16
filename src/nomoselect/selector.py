"""
GeometricSubspaceSelector — sklearn-compatible task-aware subspace selection.

Keeps the structure you care about, not just the variance that happens to be
large.  Replaces PCA with a task-first projection and exact diagnostics.

Pipeline
--------
1. Standardise features (zero mean, unit variance).
2. Reduce to the S_W eigenspace via the dual trick (essential when p >> n).
3. Regularise S_W with a Tikhonov floor: max(lambda_i, floor * lambda_max).
4. Compute S_W^{-1/2} and the whitened task family.
5. Call nomogeo to find the optimal low-dimensional view.
6. Lift back to the original feature space.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

import nomogeo  # 0.3.2 API: closure_adapted_observer, closure_scores (adapted.py)
# ── nomogeo 0.3.3 update safety note ──
# This module uses only adapted.py APIs (closure_adapted_observer,
# closure_scores). The 0.3.3 update does NOT alter adapted.py signatures.
# No changes needed here for the 0.3.3 upgrade.

from .tasks import (
    TaskFamily,
    fisher_task_family,
    equal_weight_task_family,
    minority_emphasis_family,
    pairwise_task_family,
)

_VALID_TASKS = ("fisher", "equal_weight", "minority", "pairwise")
_VALID_SOLVERS = ("commuting_exact", "eigenvector_fallback")


class GeometricSubspaceSelector(BaseEstimator, TransformerMixin):
    """Task-aware subspace selection via observer geometry.

    Finds the lowest-rank projection that preserves the structure you
    declared, and provides exact diagnostics showing what was kept and
    what was lost.

    Parameters
    ----------
    n_components : int, default 2
        Number of dimensions to keep.
    task : str or TaskFamily, default "fisher"
        What structure to preserve.  Built-in options:

        - ``"fisher"`` — sample-weighted class separation (equivalent to
          Fisher / LDA when regularisation is matched).
        - ``"equal_weight"`` — equal importance per class (good when
          class sizes are very unbalanced).
        - ``"minority"`` — inverse-frequency weighting (emphasises rare
          classes).
        - ``"pairwise"`` — one task per class pair.

        A pre-built ``TaskFamily`` can also be passed directly.
    reg_floor_frac : float, default 0.01
        Regularisation strength.  Within-class scatter eigenvalues are
        floored at ``reg_floor_frac × max_eigenvalue``.
    max_reduced_rank : int or None, default None
        Maximum dimension for the internal eigenspace reduction.
        If None, ``min(n - k, p)`` is used automatically.
    standardise : bool, default True
        Whether to centre and scale features before fitting.
    solver : str, default "commuting_exact"
        Internal solver.  ``"commuting_exact"`` uses the exact nomogeo
        solver; ``"eigenvector_fallback"`` uses top eigenvectors of the
        aggregate task matrix.

    Attributes
    ----------
    components_ : ndarray of shape (n_features, n_components)
        Projection matrix.  Use ``transform(X)`` to project new data.
    n_features_in_ : int
        Number of features seen during fit.
    closure_scores_ : ClosureScoresResult
        Exact diagnostics (retained, lost, hidden fraction).
    task_family_ : TaskFamily
        The task family used for fitting.
    eigvals_sw_ : ndarray
        Within-class scatter eigenvalues (before regularisation).
    n_above_floor_ : int
        Number of scatter eigenvalues above the regularisation floor.
    reg_floor_ : float
        Absolute regularisation floor used.
    method_ : str
        Which solver was actually used.
    """

    def __init__(
        self,
        n_components: int = 2,
        task: str | TaskFamily = "fisher",
        reg_floor_frac: float = 0.01,
        max_reduced_rank: int | None = None,
        standardise: bool = True,
        solver: Literal["commuting_exact", "eigenvector_fallback"] = "commuting_exact",
    ):
        self.n_components = n_components
        self.task = task
        self.reg_floor_frac = reg_floor_frac
        self.max_reduced_rank = max_reduced_rank
        self.standardise = standardise
        self.solver = solver

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeometricSubspaceSelector":
        """Fit the selector to labelled training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        # ── Validate inputs ──
        X, y, n, p, k, y_enc = self._validate_fit_inputs(X, y)
        self.n_features_in_ = p

        # ── Standardise ──
        if self.standardise:
            self._scaler = StandardScaler().fit(X)
            X_std = self._scaler.transform(X)
        else:
            self._scaler = None
            X_std = X.copy()

        # ── Check for constant / near-constant features after scaling ──
        col_var = np.var(X_std, axis=0)
        n_const = int(np.sum(col_var < 1e-12))
        if n_const > 0:
            warnings.warn(
                f"{n_const} feature(s) have near-zero variance after "
                f"standardisation and may contribute noise rather than "
                f"signal.  Consider removing constant features before "
                f"fitting.",
                UserWarning,
                stacklevel=2,
            )

        # ── Dual trick: reduce to S_W eigenspace ──
        max_rank = self.max_reduced_rank or min(n - k, p)
        max_rank = min(max_rank, n - k, p)
        if max_rank < 1:
            raise ValueError(
                f"Not enough samples for the number of classes.  After "
                f"reserving one sample per class, {n - k} degrees of "
                f"freedom remain for within-class scatter, which is not "
                f"enough.  You need at least {k + 1} samples for {k} "
                f"classes."
            )

        X_red, V_reduce, eigvals_sw = self._reduce_to_eigenspace(
            X_std, y_enc, k, max_rank
        )
        r = X_red.shape[1]

        # ── n_components vs usable rank ──
        if self.n_components > r:
            raise ValueError(
                f"n_components={self.n_components} is larger than the "
                f"usable rank ({r}).  The data has {k} classes and {p} "
                f"features, giving at most {r} useful discriminant "
                f"dimensions.  Try n_components <= {r}."
            )

        # ── Tikhonov regularisation ──
        if eigvals_sw[0] <= 0:
            raise ValueError(
                "Within-class scatter is numerically zero.  This usually "
                "means all samples within each class are identical (or "
                "nearly so).  There is no within-class variation to "
                "regularise against."
            )
        floor = self.reg_floor_frac * eigvals_sw[0]
        eigvals_reg = np.maximum(eigvals_sw, floor)
        self.n_above_floor_ = int(np.sum(eigvals_sw > floor))
        self.reg_floor_ = float(floor)
        self.eigvals_sw_ = eigvals_sw

        # ── Guard against zero regularised eigenvalues ──
        if np.any(eigvals_reg <= 0):
            raise ValueError(
                "Some within-class scatter eigenvalues are zero even "
                "after regularisation.  This happens when "
                "reg_floor_frac=0 and the scatter is rank-deficient.  "
                "Try setting reg_floor_frac > 0 (e.g. 0.01)."
            )

        # ── Warn on heavy regularisation ──
        frac_floored = 1.0 - self.n_above_floor_ / len(eigvals_sw)
        if frac_floored > 0.8:
            warnings.warn(
                f"{frac_floored:.0%} of within-class scatter eigenvalues "
                f"are below the regularisation floor.  The result may be "
                f"dominated by regularisation rather than data structure.  "
                f"Consider reducing reg_floor_frac (currently "
                f"{self.reg_floor_frac}) or gathering more samples.",
                UserWarning,
                stacklevel=2,
            )

        # ── Within-class scatter operators ──
        S_W_inv_sqrt_diag = 1.0 / np.sqrt(eigvals_reg)
        S_W_sqrt_diag = np.sqrt(eigvals_reg)

        # ── Class means in whitened-reduced space ──
        mu_all = X_red.mean(axis=0)
        class_means_w = np.zeros((k, r))
        class_sizes = np.zeros(k)
        for c in range(k):
            mask = y_enc == c
            class_sizes[c] = mask.sum()
            mc = X_red[mask].mean(axis=0)
            class_means_w[c] = S_W_inv_sqrt_diag * (mc - mu_all)

        # ── Warn on condition number ──
        cond = eigvals_sw[0] / max(eigvals_sw[-1], 1e-300)
        if cond > 1e8:
            warnings.warn(
                f"Within-class scatter condition number is {cond:.1e}, "
                f"which is very high.  Results may be numerically "
                f"sensitive.  The regularisation floor helps, but you "
                f"may want to check stability with "
                f"RegularisationAudit.run().",
                UserWarning,
                stacklevel=2,
            )

        # ── Build task family ──
        labels = [str(c) for c in self.classes_]
        if isinstance(self.task, TaskFamily):
            self._validate_custom_task_family(self.task, r)
            self.task_family_ = self.task
        elif self.task == "fisher":
            self.task_family_ = fisher_task_family(class_means_w, class_sizes, labels)
        elif self.task == "equal_weight":
            self.task_family_ = equal_weight_task_family(class_means_w, class_sizes, labels)
        elif self.task == "minority":
            self.task_family_ = minority_emphasis_family(class_means_w, class_sizes, labels)
        elif self.task == "pairwise":
            self.task_family_ = pairwise_task_family(class_means_w, class_sizes, labels)
        else:
            raise ValueError(
                f"Unknown task '{self.task}'.  "
                f"Use one of {_VALID_TASKS!r}, or pass a TaskFamily object."
            )

        # ── Warn on weak per-task contribution ──
        # (uses task_family_.labels, not class labels — safe for pairwise)
        if not isinstance(self.task, TaskFamily):
            total_norm = sum(
                float(np.linalg.norm(t, "fro"))
                for t in self.task_family_.individual
            )
            if total_norm > 0:
                for i, t in enumerate(self.task_family_.individual):
                    contrib = float(np.linalg.norm(t, "fro")) / total_norm
                    if contrib < 0.01:
                        warnings.warn(
                            f"Task '{self.task_family_.labels[i]}' "
                            f"contributes less than 1% of total task "
                            f"curvature under the '{self.task}' weighting.  "
                            f"This component may be invisible in the "
                            f"selected subspace.  Consider using "
                            f"task='equal_weight' or task='minority' "
                            f"to give it more influence.",
                            UserWarning,
                            stacklevel=2,
                        )

        # ── Observer selection ──
        H = np.eye(r)
        family = self.task_family_.aggregate

        if self.solver == "commuting_exact":
            try:
                result = nomogeo.closure_adapted_observer(
                    H, family, self.n_components, mode="commuting_exact"
                )
                B_w = result.B
                self.method_ = "commuting_exact"
            except Exception as exc:
                warnings.warn(
                    f"The exact solver failed and the eigenvector fallback "
                    f"was used instead.  The results are still valid but "
                    f"may not be globally optimal.  "
                    f"(Solver error: {exc})",
                    UserWarning,
                    stacklevel=2,
                )
                B_w = self._eigenvector_fallback(family[0], self.n_components)
                self.method_ = "eigenvector_fallback"
        else:
            B_w = self._eigenvector_fallback(family[0], self.n_components)
            self.method_ = "eigenvector_fallback"

        # ── Closure diagnostics ──
        self.closure_scores_ = nomogeo.closure_scores(H, family, B_w)

        # ── Lift to original standardised space ──
        B_reduced = S_W_inv_sqrt_diag[:, None] * B_w
        B_full = V_reduce @ B_reduced
        B_full, _ = np.linalg.qr(B_full)
        self.components_ = B_full

        # ── Store internals for report / audit ──
        self._V_reduce = V_reduce
        self._S_W_inv_sqrt_diag = S_W_inv_sqrt_diag
        self._S_W_sqrt_diag = S_W_sqrt_diag
        self._B_whitened = B_w
        self._H = H
        self._r = r
        self._class_means_w = class_means_w
        self._class_sizes = class_sizes

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the selected subspace.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        X_proj : ndarray of shape (n_samples, n_components)
        """
        check_is_fitted(self, "components_")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}."
            )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the selector was "
                f"fitted on {self.n_features_in_} features."
            )
        if np.any(~np.isfinite(X)):
            n_bad = int(np.sum(~np.isfinite(X)))
            raise ValueError(
                f"X contains {n_bad} non-finite value(s) (NaN or Inf).  "
                f"Clean your data before transforming."
            )
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X @ self.components_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and project in one step."""
        return self.fit(X, y).transform(X)

    # ── Input validation ─────────────────────────────────────────────

    def _validate_fit_inputs(self, X, y):
        """Validate and sanitise inputs to fit().  Returns cleaned arrays."""
        # Convert to numpy
        try:
            X = np.asarray(X, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"X could not be converted to a numeric array.  "
                f"Check for non-numeric or missing values.  "
                f"(Original error: {e})"
            ) from e

        y = np.asarray(y)

        # Shape checks
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array (n_samples x n_features), "
                f"got shape {X.shape}."
            )
        n, p = X.shape
        if n < 2:
            raise ValueError(
                f"Need at least 2 samples to fit, got {n}."
            )
        if p < 1:
            raise ValueError("X has no features.")
        if y.ndim != 1 or len(y) != n:
            raise ValueError(
                f"y must be a 1-D array with {n} elements (one per "
                f"sample), got shape {y.shape}."
            )

        # NaN / Inf checks
        if np.any(~np.isfinite(X)):
            n_bad = int(np.sum(~np.isfinite(X)))
            raise ValueError(
                f"X contains {n_bad} non-finite value(s) (NaN or Inf).  "
                f"Remove or impute them before fitting."
            )

        # Label checks
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        self.classes_ = list(le.classes_)
        k = len(self.classes_)

        if k < 2:
            raise ValueError(
                f"Found only {k} class(es) in y.  Task-aware subspace "
                f"selection requires at least 2 distinct classes."
            )

        # Per-class sample count
        for c in range(k):
            nc = int((y_enc == c).sum())
            if nc < 2:
                raise ValueError(
                    f"Class '{self.classes_[c]}' has only {nc} sample(s).  "
                    f"Every class needs at least 2 samples so that "
                    f"within-class scatter can be estimated."
                )

        # n_components checks
        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError(
                f"n_components must be an integer, got "
                f"{type(self.n_components).__name__}."
            )
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1, got {self.n_components}."
            )
        max_useful = min(k - 1, p)
        if self.n_components > max_useful:
            raise ValueError(
                f"n_components={self.n_components} is too large.  With "
                f"{k} classes and {p} features, there are at most "
                f"{max_useful} useful discriminant dimensions.  "
                f"Try n_components <= {max_useful}."
            )

        # reg_floor_frac checks
        if not isinstance(self.reg_floor_frac, (int, float, np.floating)):
            raise TypeError(
                f"reg_floor_frac must be a number, got "
                f"{type(self.reg_floor_frac).__name__}."
            )
        if self.reg_floor_frac < 0 or self.reg_floor_frac >= 1:
            raise ValueError(
                f"reg_floor_frac must be in [0, 1), got "
                f"{self.reg_floor_frac}.  Typical values are 0.001 to 0.1."
            )

        # max_reduced_rank checks
        if self.max_reduced_rank is not None:
            if not isinstance(self.max_reduced_rank, (int, np.integer)):
                raise TypeError(
                    f"max_reduced_rank must be an integer or None, got "
                    f"{type(self.max_reduced_rank).__name__}."
                )
            if self.max_reduced_rank < 1:
                raise ValueError(
                    f"max_reduced_rank must be at least 1, got "
                    f"{self.max_reduced_rank}."
                )

        # solver check
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(
                f"Unknown solver '{self.solver}'.  "
                f"Use one of {_VALID_SOLVERS!r}."
            )

        return X, y, n, p, k, y_enc

    @staticmethod
    def _validate_custom_task_family(tf: TaskFamily, r: int):
        """Validate a user-supplied TaskFamily against the reduced dimension."""
        # Aggregate shape
        for i, agg in enumerate(tf.aggregate):
            agg = np.asarray(agg)
            if agg.shape != (r, r):
                raise ValueError(
                    f"Custom TaskFamily aggregate[{i}] has shape "
                    f"{agg.shape}, but the reduced dimension is {r}.  "
                    f"Expected ({r}, {r}).  Make sure the task matrices "
                    f"match the whitened feature space."
                )
        # Individual shape
        for i, ind in enumerate(tf.individual):
            ind = np.asarray(ind)
            if ind.shape != (r, r):
                raise ValueError(
                    f"Custom TaskFamily individual[{i}] has shape "
                    f"{ind.shape}, expected ({r}, {r}).  "
                    f"Task matrices must match the whitened feature space."
                )
        # Weights length
        if len(tf.weights) != len(tf.individual):
            raise ValueError(
                f"Custom TaskFamily has {len(tf.weights)} weights but "
                f"{len(tf.individual)} individual tasks.  These must "
                f"match."
            )
        # Labels length
        if len(tf.labels) != len(tf.individual):
            raise ValueError(
                f"Custom TaskFamily has {len(tf.labels)} labels but "
                f"{len(tf.individual)} individual tasks.  These must "
                f"match."
            )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _reduce_to_eigenspace(
        X_std: np.ndarray, y: np.ndarray, k: int, max_rank: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dual trick: project to the S_W eigenspace via SVD of residuals.

        Returns
        -------
        X_red : (n, r) reduced data
        V : (p, r) projection matrix
        eigvals : (r,) eigenvalues of S_W, sorted descending
        """
        n, p = X_std.shape
        classes = np.unique(y)
        residuals = np.zeros_like(X_std)
        for c in classes:
            mask = y == c
            residuals[mask] = X_std[mask] - X_std[mask].mean(axis=0)

        U, s, Vt = np.linalg.svd(residuals, full_matrices=False)
        rank = min(max_rank, len(s))
        s = s[:rank]
        V = Vt[:rank].T
        eigvals = s ** 2 / n
        X_red = X_std @ V
        return X_red, V, eigvals

    @staticmethod
    def _eigenvector_fallback(agg_matrix: np.ndarray, rank: int) -> np.ndarray:
        """Top eigenvectors of the aggregate task matrix."""
        evals, evecs = np.linalg.eigh(agg_matrix)
        return evecs[:, -rank:]
