"""
Automated case-study runner for nomoselect v0.2.0.

Runs GeometricSubspaceSelector across 20 classification scenarios,
compares against PCA at each rank, and collects structured results.

Output: case_study_results.json  (machine-readable)
        case_study_summary.txt   (human-readable)
"""
import json
import sys
import os
import time
import warnings

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# ── Path setup (works from workspace checkout) ──
_here = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_here)
sys.path.insert(0, os.path.join(_workspace, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_workspace), "observer_geometry", "src"))

from nomoselect import (
    GeometricSubspaceSelector,
    ObserverReport,
    RegularisationAudit,
    DimensionCostLadder,
)

warnings.filterwarnings("ignore")

# ── Dataset catalogue ──────────────────────────────────────────────


def _make_synth(n, p, k, informative, redundant, sep, flip, weights, seed, name):
    """Wrapper for make_classification with structured metadata."""
    w = None if weights is None else weights[:k]
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=informative,
        n_redundant=redundant, n_classes=k, n_clusters_per_class=1,
        class_sep=sep, flip_y=flip, weights=w,
        random_state=seed,
    )
    return X, y, name


DATASETS = []

# ── Real datasets ──
def _add_real(loader, name):
    d = loader()
    DATASETS.append((d.data, d.target, name))

_add_real(load_iris, "Iris (4d, 3 class)")
_add_real(load_wine, "Wine (13d, 3 class)")
_add_real(load_breast_cancer, "Breast Cancer (30d, 2 class)")
_add_real(load_digits, "Digits (64d, 10 class)")

# ── Synthetic: controlled scenarios ──
DATASETS.append(_make_synth(200, 10, 3, 5, 2, 1.5, 0.0, None, 1,
    "Synth: clean 3-class (10d)"))
DATASETS.append(_make_synth(200, 10, 3, 5, 2, 0.5, 0.0, None, 2,
    "Synth: low-sep 3-class (10d)"))
DATASETS.append(_make_synth(200, 10, 5, 5, 2, 1.0, 0.0, None, 3,
    "Synth: 5-class (10d)"))
DATASETS.append(_make_synth(200, 50, 3, 5, 10, 1.0, 0.0, None, 4,
    "Synth: noisy 3-class (50d)"))
DATASETS.append(_make_synth(200, 50, 3, 3, 5, 1.0, 0.0, None, 5,
    "Synth: few-informative (50d)"))
DATASETS.append(_make_synth(300, 100, 4, 10, 20, 1.0, 0.0, None, 6,
    "Synth: moderate p (100d, 4 class)"))
DATASETS.append(_make_synth(100, 200, 3, 8, 10, 1.0, 0.0, None, 7,
    "Synth: p>n (200d, 3 class, n=100)"))
DATASETS.append(_make_synth(60, 500, 3, 10, 20, 1.5, 0.0, None, 8,
    "Synth: high-p (500d, n=60)"))
DATASETS.append(_make_synth(200, 20, 2, 5, 5, 2.0, 0.0, None, 9,
    "Synth: binary easy (20d)"))
DATASETS.append(_make_synth(200, 20, 2, 3, 5, 0.3, 0.1, None, 10,
    "Synth: binary hard + noise (20d)"))
DATASETS.append(_make_synth(300, 10, 3, 5, 2, 1.0, 0.0, [0.1, 0.2, 0.7], 11,
    "Synth: imbalanced 3-class"))
DATASETS.append(_make_synth(300, 10, 4, 4, 2, 1.0, 0.0, [0.05, 0.15, 0.3, 0.5], 12,
    "Synth: heavily imbalanced 4-class"))
DATASETS.append(_make_synth(500, 30, 6, 10, 5, 0.8, 0.0, None, 13,
    "Synth: 6-class moderate (30d)"))
DATASETS.append(_make_synth(200, 10, 3, 10, 0, 2.0, 0.0, None, 14,
    "Synth: all-informative (10d, 3 class)"))
DATASETS.append(_make_synth(150, 40, 3, 5, 10, 1.0, 0.05, None, 15,
    "Synth: label noise 5% (40d)"))
DATASETS.append(_make_synth(150, 40, 3, 5, 10, 1.0, 0.15, None, 16,
    "Synth: label noise 15% (40d)"))


# ── Runner ─────────────────────────────────────────────────────────


def run_case(X, y, name, tasks=("fisher", "equal_weight")):
    """Run one case study.  Returns a results dict."""
    n, p = X.shape
    k = len(np.unique(y))
    max_rank = min(k - 1, 10, p)

    # Standardise once for PCA baseline
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)

    result = {
        "name": name,
        "n": n, "p": p, "k": k,
        "max_rank": max_rank,
        "tasks": {},
    }

    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)

    for task_name in tasks:
        task_result = {"ranks": {}}

        for rank in range(1, max_rank + 1):
            try:
                # GEO
                sel = GeometricSubspaceSelector(
                    n_components=rank, task=task_name
                ).fit(X, y)
                X_geo = sel.transform(X)

                sc = sel.closure_scores_
                pct_retained = 100.0 * sc.visible_score / max(sc.total_curvature, 1e-300)

                # PCA at same rank
                pca = PCA(n_components=rank).fit(X_std)
                X_pca = X_std @ pca.components_.T

                # CV accuracy comparison (KNN k=3)
                n_splits = min(5, min(np.bincount(y)))
                if n_splits >= 2:
                    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    acc_geo = float(cross_val_score(
                        KNeighborsClassifier(n_neighbors=min(3, n-1)),
                        X_geo, y, cv=cv_inner, scoring="accuracy"
                    ).mean())
                    acc_pca = float(cross_val_score(
                        KNeighborsClassifier(n_neighbors=min(3, n-1)),
                        X_pca, y, cv=cv_inner, scoring="accuracy"
                    ).mean())
                else:
                    acc_geo = acc_pca = float("nan")

                # Dominance check via report
                B_pca_full = pca.components_.T  # (p, rank)
                report = ObserverReport.from_selector(sel, baselines={"PCA": B_pca_full})
                if report.baselines:
                    bc = report.baselines[0]
                    geo_dominates = bc.geo_dominates
                    pca_dominates = bc.baseline_dominates
                else:
                    geo_dominates = pca_dominates = False

                task_result["ranks"][str(rank)] = {
                    "pct_retained": round(pct_retained, 2),
                    "eta": round(sc.eta, 4),
                    "acc_geo": round(acc_geo, 4),
                    "acc_pca": round(acc_pca, 4),
                    "acc_delta": round(acc_geo - acc_pca, 4),
                    "geo_dominates": geo_dominates,
                    "pca_dominates": pca_dominates,
                    "method": sel.method_,
                }
            except Exception as e:
                task_result["ranks"][str(rank)] = {"error": str(e)}

        # Stability audit at rank 1
        try:
            audit = RegularisationAudit.run(
                X, y, n_components=1, task=task_name,
                floor_fracs=[0.001, 0.01, 0.1],
            )
            task_result["stable"] = audit.stable
        except Exception:
            task_result["stable"] = None

        # Dimension ladder
        try:
            ladder = DimensionCostLadder.build(X, y, task=task_name)
            task_result["recommended_rank"] = ladder.recommended_rank
        except Exception:
            task_result["recommended_rank"] = None

        result["tasks"][task_name] = task_result

    return result


def main():
    print(f"Running {len(DATASETS)} case studies...")
    all_results = []
    t0 = time.time()

    for i, (X, y, name) in enumerate(DATASETS):
        print(f"  [{i+1:2d}/{len(DATASETS)}] {name}...", end=" ", flush=True)
        t1 = time.time()
        r = run_case(X, y, name)
        dt = time.time() - t1
        print(f"done ({dt:.1f}s)")
        all_results.append(r)

    total = time.time() - t0
    print(f"\nAll done in {total:.1f}s")

    # Save JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "case_study_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results: {json_path}")

    # Generate summary
    summary_path = os.path.join(out_dir, "case_study_summary.txt")
    with open(summary_path, "w") as f:
        f.write("nomoselect v0.2.0 — Case Study Results\n")
        f.write("=" * 60 + "\n\n")

        geo_wins = 0
        pca_wins = 0
        ties = 0
        total_cases = 0

        for r in all_results:
            f.write(f"{r['name']}  (n={r['n']}, p={r['p']}, k={r['k']})\n")
            for task_name, task_data in r["tasks"].items():
                f.write(f"  task={task_name}")
                if task_data.get("stable") is not None:
                    f.write(f"  stable={task_data['stable']}")
                if task_data.get("recommended_rank") is not None:
                    f.write(f"  rec_rank={task_data['recommended_rank']}")
                f.write("\n")

                for rank_str, rd in sorted(task_data.get("ranks", {}).items()):
                    rank = int(rank_str)
                    if "error" in rd:
                        f.write(f"    rank {rank}: ERROR {rd['error']}\n")
                        continue

                    delta = rd["acc_delta"]
                    dom = ""
                    if rd["geo_dominates"]:
                        dom = " [GEO dominates]"
                        geo_wins += 1
                    elif rd["pca_dominates"]:
                        dom = " [PCA dominates]"
                        pca_wins += 1
                    else:
                        dom = ""
                        ties += 1
                    total_cases += 1

                    f.write(
                        f"    rank {rank}: "
                        f"retained={rd['pct_retained']:5.1f}%  "
                        f"GEO={rd['acc_geo']:.3f}  "
                        f"PCA={rd['acc_pca']:.3f}  "
                        f"delta={delta:+.3f}"
                        f"{dom}\n"
                    )
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"Dominance summary across all rank/task combinations:\n")
        f.write(f"  GEO dominates: {geo_wins}/{total_cases}\n")
        f.write(f"  PCA dominates: {pca_wins}/{total_cases}\n")
        f.write(f"  Neither:       {ties}/{total_cases}\n")
        f.write("\n")
        f.write("Note: dominance is an exact certificate on task-curvature,\n")
        f.write("not a claim about predictive accuracy.  Accuracy deltas\n")
        f.write("depend on the classifier and the task declaration.\n")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
