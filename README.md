# Symbolic Regression with Genetic Programming

Genetic Programming (GP) symbolic regression that searches for closed-form expressions of a material's **heat of formation** from elemental-property descriptors, with an MLP baseline for comparison.

## Contents

| Path | Description |
|---|---|
| `gp/` | GP engine, a modified copy of [gplearn](https://github.com/trevorstephens/gplearn) (`SymbolicRegressor`, `SymbolicClassifier`, `SymbolicTransformer`) |
| `data_utils.py` | Loads a dataset, splits it into train/test, and standardizes the features. The scaler is fit on the training set only |
| `GP_example.ipynb` | Example on the `high` dataset: a first GP model, a second GP model fit to its residuals, and the combined performance |
| `MLP_regression.py` | MLP baseline (`high`, `mid`, `low`) |
| `dataset/` | CSV data |
| `tests/` | pytest suite |

## Changes from gplearn

- Adds `pow2` (square) and `pow3` (cube), and extends the default `function_set` of `SymbolicRegressor` to 14 functions
- Programs that overflow or produce NaN get the worst possible fitness, so `np.argmin`/`np.argmax` can no longer select them. `parsimony_coefficient='auto'` is computed from finite values only
- Compatible with NumPy 2.x and scikit-learn 1.6+: no `np.int` or `numpy.lib.arraysetops`, mixins placed before `BaseEstimator`, classifier tags provided through `__sklearn_tags__`
- Importable as `from gp import SymbolicRegressor` (`gp/__init__.py`) and installable with pip (`pyproject.toml`)

## Installation

To use it from another project's environment (requires numpy, scipy, scikit-learn, joblib):

```bash
pip install -e /path/to/Symbolic_regression
```

To create a dedicated environment with the tested versions pinned:

```bash
conda env create -f environment.yml
conda activate symreg
```

## Usage

```python
import numpy as np
from gp import SymbolicRegressor

rng = np.random.RandomState(42)
X = rng.uniform(1, 5, (500, 3))
y = X[:, 0] * X[:, 1] / X[:, 2]

est = SymbolicRegressor(population_size=3000, generations=20,
                        function_set=('add', 'sub', 'mul', 'div'),
                        parsimony_coefficient=0.001,
                        feature_names=['P', 'V', 'T'],
                        random_state=0, n_jobs=4)
est.fit(X, y)
print(est)  # div(mul(P, V), T)
```

- Create custom functions with `gp.make_function` and custom fitness measures with `gp.make_fitness`
- The estimators follow the scikit-learn API, so `Pipeline`, `cross_val_score`, `clone`, and pickling work as usual
- With the same `random_state`, results are identical regardless of `n_jobs`

## Data

`data_utils.load_split(name, target='Heat of formation')` returns `x_tr, x_ts, y_tr, y_ts, feature_names`. Every column except `index_col` (default `Material`) and `target` becomes a feature, so with the defaults `Energy above convex hull` is also used as a feature (85 features). The data folder is resolved in this order:

1. The `data_dir` argument
2. `$DATA_ROOT/symbolic_regression`, if that folder exists
3. `dataset/` in this repository

| File | Rows | Columns | Contents |
|---|---|---|---|
| `high.csv` | 1,366 | 87 | `Material` + 84 descriptors (`Z_Average`, `Z_Weighted_Average`, `Z_Max`, …) + `Heat of formation` + `Energy above convex hull` |
| `mid.csv` | 590 | 87 | Same columns as `high.csv` |
| `low.csv` | 469 | 87 | Same columns as `high.csv` |
| `crawl.csv` | 461 | 87 | Same columns as `high.csv` |
| `Schleder2019_AtomicTable.csv` | 118 | 18 | Elemental property table (`Element`, `Z`, `Electronegativity`, …) |

## Running

Run from the repository root:

```bash
python -m pytest
python MLP_regression.py
jupyter nbconvert --to notebook --execute --inplace GP_example.ipynb
```

## Changelog

- 2026-09: Fixed the following bugs, so the notebook and MLP numbers may differ from the original 2021 runs
  - The scaler was re-fit on the test set (`fit_transform` → `transform`)
  - Argument order of `r2_score` in `MLP_regression.py`
  - `init_depth` was not passed to the first GP run in the notebook

## Tested environment

2026-09, WSL2 Ubuntu: Python 3.11.15, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, pandas 3.0.3, joblib 1.5.3.

## License

- `gp/` is based on gplearn (BSD 3-Clause, Copyright (c) 2015, Trevor Stephens). Include `gp/LICENSE` when redistributing
- No license has been chosen yet for the other files
