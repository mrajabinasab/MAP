# MAP: Mean Angular Preservation

MAP computes a normalized **Mean Angular Preservation** score for a set of selected features. It measures how much the first principal component changes when each non-selected feature is set to zero in the standardized data.

This implementation provides two execution modes:

* **Direct mode**: computes only the values needed for the current call.
* **Benchmark/cache mode**: precomputes and caches per-feature values for faster repeated calls on the same dataset.

All per-feature values are computed exactly from a single singular value decomposition of the standardized data, without fitting one PCA per feature.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The package depends on:

* `numpy`
* `pandas`
* `scikit-learn`

## Usage

```python
import pandas as pd
from map import MAP

X = pd.DataFrame({
    "a": [1.0, 2.0, 3.0, 4.0],
    "b": [2.0, 1.0, 4.0, 3.0],
    "c": [5.0, 6.0, 7.0, 8.0],
})

score = MAP(X, selected_features=["a", "b"])

print(score)
```

You can also pass integer column indices instead of column names:

```python
score = MAP(X, selected_features=[0, 1])
```

## Cached / benchmark mode

When calling `MAP` many times on the same dataset, set `benchmark=True` to cache the expensive shared calculations:

```python
from map import MAP, clear_map_cache

score_1 = MAP(X, selected_features=["a"], benchmark=True)
score_2 = MAP(X, selected_features=["a", "b"], benchmark=True)
score_3 = MAP(X, selected_features=["b", "c"], benchmark=True)
```

This is useful when evaluating many candidate feature subsets against the same `X`.

## Cache invalidation

The cache is tied to the identity of the input object. If you modify `X` in place, rebuild the cache:

```python
score = MAP(
    X,
    selected_features=["a", "b"],
    benchmark=True,
    force_rebuild=True,
)
```

Or clear all cached values:

```python
clear_map_cache()
```

## API

### `MAP`

```python
MAP(
    X: pd.DataFrame,
    selected_features: list,
    benchmark: bool = False,
    force_rebuild: bool = False,
) -> float
```

Compute the MAP score.

#### Parameters

| Parameter           |           Type |  Default | Description                                                                                                       |
| ------------------- | -------------: | -------: | ----------------------------------------------------------------------------------------------------------------- |
| `X`                 | `pd.DataFrame` | required | Input feature matrix. A DataFrame is recommended, especially when using string feature names.                     |
| `selected_features` |         `list` | required | Selected feature names or integer column indices.                                                                 |
| `benchmark`         |         `bool` |  `False` | If `False`, computes MAP directly. If `True`, caches per-feature delta values for repeated calls on the same `X`. |
| `force_rebuild`     |         `bool` |  `False` | If `True` and `benchmark=True`, rebuilds the cache for `X`.                                                       |

#### Returns

A normalized MAP score in `[0, 1]` as a `float`.

In this implementation, if all features are selected, `MAP` returns `1`.

### `clear_map_cache`

```python
clear_map_cache() -> None
```

Clear all cached MAP computations.

## Example: feature subset comparison

```python
candidate_sets = [
    ["a"],
    ["a", "b"],
    ["a", "c"],
]

scores = {
    tuple(features): MAP(X, features, benchmark=True)
    for features in candidate_sets
}

print(scores)
```

## Citation

Citation will be provided upon publication.
