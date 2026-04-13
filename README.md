# nomoselect

Task-aware dimensionality reduction for classification.

PCA keeps what is biggest.  
nomoselect keeps what matters for the task you care about.

## What it does

nomoselect finds a low-dimensional view of your data that is built for a declared classification goal.

In simple terms, it answers this question:

**If I only keep 1, 2, or 3 dimensions, which view of the data keeps the class structure I care about most?**

It also tells you, in plain language:

- how much of that target structure was kept
- how much was lost
- whether the result is stable across regularisation choices
- whether another dimension is worth paying for

This makes it useful when ordinary dimensionality reduction feels too generic.

PCA is a safe default when you just want a compact summary of the data.  
nomoselect is for the different situation: **you already know what structure matters, and you want the low-dimensional view to preserve that structure on purpose.**

## What it is for

nomoselect is designed for supervised dimensionality reduction.

Typical use cases include:

- reducing a high-dimensional dataset before classification
- keeping minority-class structure visible at low rank
- choosing how many dimensions to keep in a task-aware way
- auditing whether a projection is actually aligned with the target
- making regularisation choices visible instead of hidden inside solver defaults

## What it is not

nomoselect is not a general replacement for PCA in every workflow.

If you just want an unsupervised variance summary, PCA is still the simpler tool.

nomoselect is for cases where you want the low-dimensional representation to be tied to a declared task, and where you want feedback about what that choice kept and what it threw away.

## Install

```bash
pip install nomoselect
````

Requires:

* Python >= 3.10
* numpy
* scikit-learn
* nomogeo >= 0.3.2

## Quickstart

```python
from sklearn.datasets import load_iris
from nomoselect import GeometricSubspaceSelector, ObserverReport

X, y = load_iris(return_X_y=True)

# Keep 2 dimensions, chosen to preserve class separation
sel = GeometricSubspaceSelector(
    n_components=2,
    task="fisher",
)

X_proj = sel.fit_transform(X, y)

report = ObserverReport.from_selector(sel)
print(report.summary())
```

Example output:

```text
nomoselect report  (2 dimensions kept)

  Target structure retained : 100.0%
  Target structure lost     :   0.0%

  Per-class retention:
    0                     retained 100.0%  (weight=0.333)
    1                     retained 100.0%  (weight=0.333)
    2                     retained 100.0%  (weight=0.333)

  Note: these diagnostics are exact for the task you
  declared. They do not prove the task itself is valid.
  Cross-validation or permutation tests are still needed
  to confirm that the structure is real.
```

## How to read the output

The summaries are meant to be readable without needing the internal theory.

**Target structure retained**
How much of the class structure you asked nomoselect to preserve is still visible in the chosen projection.

**Target structure lost**
How much of that structure falls outside the chosen projection.

**Per-class retention**
How well each class contributes to the chosen low-dimensional view. This is useful for spotting when a minority class has been pushed aside by a majority-weighted objective.

The final note matters:

nomoselect tells you whether the projection is good **for the task you declared**.
It does **not** tell you whether that task is scientifically valid on its own.
That still needs external checks such as cross-validation, held-out testing, or permutation tests.

## Choosing a task

A task tells nomoselect what kind of class structure should matter most.

```python
from nomoselect import GeometricSubspaceSelector

sel = GeometricSubspaceSelector(n_components=2, task="fisher")
```

Built-in task families:

| Task             | What it means in simple terms                            | When to use                                                         |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| `"fisher"`       | Preserve class separation with ordinary sample weighting | Default choice; matches Fisher/LDA when regularisation is matched   |
| `"equal_weight"` | Give each class the same importance                      | Useful when class sizes are uneven                                  |
| `"minority"`     | Push more rank budget toward rare classes                | Useful when missing a rare class matters more than average accuracy |
| `"pairwise"`     | Treat each class pair as its own separation problem      | Useful when fine-grained class confusion matters                    |

These are different modelling choices, not just different settings.

For example, a minority-emphasis task may improve recall on a rare class while giving up some overall accuracy. nomoselect makes that trade-off visible instead of hiding it.

## GEO, LDA, and PCA

### PCA

PCA is unsupervised.

It finds directions with large variance, whether or not that variance is relevant to your classification problem.

That makes it fast, general, and useful.
It also means PCA can spend dimensions on variation that is off-task.

### LDA

LDA is supervised.

It chooses directions that separate classes under Fisher's criterion.

When you use:

```python
task="fisher"
```

and match the regularisation, nomoselect recovers the same subspace as Fisher/LDA.

### GEO

GEO is the selection rule used by nomoselect.

When the declared task is Fisher, GEO gives Fisher.
When the declared task is different, GEO gives the best low-dimensional view for that declared task.

That is the main difference.

nomoselect does not pretend LDA is wrong.
It puts LDA inside a wider framework where you can:

* recover Fisher when Fisher is the right choice
* choose a different task when Fisher is not enough
* see what the projection kept and what it lost
* inspect regularisation instead of treating it as a hidden implementation detail
* choose rank with a task-aware alternative to a scree plot

## Why use nomoselect instead of PCA

Use nomoselect when these questions matter:

* Is my first component actually aligned with the target, or just with large irrelevant variance?
* Can I get the same task performance with fewer dimensions?
* Is a minority class being ignored by the default weighting?
* How much do I gain by adding one more dimension?
* Is my answer stable across regularisation choices?

PCA does not try to answer those questions.
nomoselect does.

## Stability and rank choice

nomoselect includes two built-in tools beyond the selector itself.

### Regularisation audit

```python
from nomoselect import RegularisationAudit

audit = RegularisationAudit.run(X, y, n_components=2)
print(audit.summary())
```

This tells you whether the chosen subspace is stable across a sweep of regularisation floors.

Use it when the dataset is high-dimensional, small-sample, or poorly conditioned.

### Dimension-cost ladder

```python
from nomoselect import DimensionCostLadder

ladder = DimensionCostLadder.build(X, y)
print(ladder.summary())
```

This is the task-aware replacement for a scree plot.

Instead of asking:

**How much variance do I gain by adding another component?**

it asks:

**How much target structure do I gain by adding another dimension?**

That gives you a more direct answer to:

**Is one more dimension worth it?**

## API overview

### Main selector

```python
from nomoselect import GeometricSubspaceSelector
```

Key parameters:

* `n_components` — number of dimensions to keep
* `task` — built-in task family or custom task object
* `reg_floor_frac` — regularisation floor as a fraction of the top within-class eigenvalue
* `solver` — selection backend

Key fitted attributes:

* `components_` — projection directions in the original feature space
* `closure_scores_` — internal retention / loss diagnostics
* `eigvals_sw_` — within-class scatter eigenvalues
* `reg_floor_` — actual regularisation floor used
* `method_` — solver path used internally
* `n_features_in_` — number of input features seen during fit

### Reporting

```python
from nomoselect import ObserverReport
```

Turns a fitted selector into a readable report, with optional comparison against a baseline such as PCA.

### Stability audit

```python
from nomoselect import RegularisationAudit
```

Sweeps the regularisation floor and reports whether the selected subspace is stable.

### Rank ladder

```python
from nomoselect import DimensionCostLadder
```

Builds a task-aware phase diagram showing when rank 1, 2, 3, ... becomes worth paying for.

## What the package guarantees

nomoselect is careful about bad inputs and edge cases.

It checks:

* shape mismatches
* missing values and infinities
* impossible rank requests
* too few classes or too few samples per class
* bad task names and invalid parameters
* misuse of `transform()` before fitting

The goal is that a careful new user can fit it, misuse it once or twice, and still understand what happened.

## Current validation cases

The package has been validated on several small and medium benchmark datasets.

The important pattern is consistent:

* when the task is Fisher, the method recovers Fisher/LDA when regularisation is matched
* against PCA, the method uses rank more efficiently for supervised tasks
* alternative task families can recover structure that ordinary Fisher weighting pushes aside
* the rank ladder gives a direct task-aware answer to whether another dimension is worth it

## When not to use it

Do not use nomoselect just because you want any low-dimensional picture at all.

Use it when you can say, in words, what structure matters.

Good examples:

* disease class
* fraud vs non-fraud
* rare class detection
* one-vs-rest separation
* pairwise class confusion
* weighted class priorities

If you cannot declare the task, PCA or another unsupervised method may still be the better first step.

## License

BSD 3-Clause License