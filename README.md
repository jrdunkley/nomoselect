# nomoselect

Task-aware dimensionality reduction with exact diagnostics.

PCA keeps what is biggest. nomoselect keeps what matters.

**Documentation:** [docs.nomogenetics.com/nomoselect](https://docs.nomogenetics.com/nomoselect.html)

## What it does

nomoselect finds the lowest-dimensional view of your data that preserves the structure you actually care about. It tells you exactly what was kept and what was lost — not as a heuristic, but as an exact certificate.

## Install

```bash
pip install nomoselect
```

Requires Python >= 3.10, numpy, scikit-learn, and [nomogeo](https://github.com/jrdunkley/nomogeo) >= 0.4.0.

## Quickstart

```python
from sklearn.datasets import load_iris
from nomoselect import GeometricSubspaceSelector, ObserverReport

X, y = load_iris(return_X_y=True)

# Fit: 2 dimensions, preserving class separation
sel = GeometricSubspaceSelector(n_components=2, task="fisher")
X_proj = sel.fit_transform(X, y)

# Diagnostics: what was kept?
report = ObserverReport.from_selector(sel)
print(report.summary())
```

Output:
```
nomoselect report  (2 dimensions kept)

  Target structure retained : 100.0%
  Target structure lost     :   0.0%

  Per-class retention:
    0                     retained 100.0%  (weight=0.333)
    1                     retained 100.0%  (weight=0.333)
    2                     retained 100.0%  (weight=0.333)

  Note: these diagnostics are exact for the task you
  declared.  They do not prove the task itself is valid.
  Cross-validation or permutation tests are still needed
  to confirm that the structure is real.
```

## Task families

nomoselect supports multiple ways to declare what structure matters:

| Task | What it preserves | When to use |
|------|-------------------|-------------|
| `"fisher"` | Sample-weighted class separation | Default; equivalent to Fisher/LDA |
| `"equal_weight"` | Equal importance per class | Unbalanced class sizes |
| `"minority"` | Inverse-frequency weighting | Rare-class detection |
| `"pairwise"` | Every class pair equally | Fine-grained separation |

## Results

Task-aware observer beats PCA consistently:

| Dataset | Advantage over PCA |
|---------|-------------------|
| Iris | +0.32 |
| Wine | +1.00 (maximum possible) |
| Breast Cancer | +0.15 |
| Digits | +0.09 |

The biggest gains appear when variance and task structure point in different directions. On well-separated data where PCA already aligns with the class boundary, both methods agree.

## Beyond selection: diagnostics

```python
from nomoselect import RegularisationAudit, DimensionCostLadder

# Is my result stable across regularisation choices?
audit = RegularisationAudit.run(X, y, n_components=2)
print(audit.summary())

# How many dimensions do I actually need?
ladder = DimensionCostLadder.build(X, y)
print(ladder.summary())
```

## How it relates to PCA and LDA

nomoselect includes Fisher/LDA as a special case. When `task="fisher"` and regularisation is matched, it recovers the same subspace as sklearn's `LinearDiscriminantAnalysis`. The difference is that nomoselect:

- tells you exactly how much target structure each dimension captures
- lets you declare alternative tasks (equal-weight, minority, pairwise)
- provides exact certificates for what was kept and what leaked
- gives you a task-aware rank selection diagram instead of a scree plot

For high-dimensional data (features >> samples), apply PCA pre-reduction first, then nomoselect.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
