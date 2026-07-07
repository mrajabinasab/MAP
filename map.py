import weakref
from dataclasses import dataclass
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class _MAPCache:
    columns: pd.Index
    delta: np.ndarray
    theta_max: float
    theta_min: float


_CACHE = {}


def clear_map_cache() -> None:
    """Clear all cached MAP computations."""
    _CACHE.clear()


def _drop_cache(key: int) -> None:
    _CACHE.pop(key, None)


def _as_dataframe(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _selected_indices(selected_features, columns: pd.Index) -> set:
    return {
        columns.get_loc(feature) if isinstance(feature, str) else feature
        for feature in selected_features
    }


def _delta_theta_max(C: np.ndarray, eigengap: float) -> float:
    max_dk_norm = 0.0

    for f in range(C.shape[0]):
        cpp = C[f, f]
        off = C[:, f].copy()
        off[f] = 0.0
        off_norm = np.linalg.norm(off)
        dk_norm = (cpp + np.sqrt(cpp**2 + 4.0 * off_norm**2)) / 2.0
        max_dk_norm = max(max_dk_norm, dk_norm)

    sin_max = np.clip(max_dk_norm / eigengap, 0.0, 1.0)
    return float((2.0 / np.pi) * np.arcsin(sin_max))


def _delta_one(p: int, X_scaled: np.ndarray, base_vector: np.ndarray) -> float:
    X_comp = X_scaled.copy()
    X_comp[:, p] = 0.0

    comp_vector = PCA(n_components=1).fit(X_comp).components_[0]
    dot = np.clip(abs(np.dot(base_vector, comp_vector)), 0.0, 1.0)
    return float((2.0 / np.pi) * np.arccos(dot))


def _shared_terms(X: pd.DataFrame):
    n_samples, n_features = X.shape

    X_scaled = StandardScaler().fit_transform(X)

    base_pca = PCA(n_components=2).fit(X_scaled)
    l1, l2 = base_pca.explained_variance_[0], base_pca.explained_variance_[1]
    base_vector = base_pca.components_[0]
    eigengap = max(l1 - l2, 1e-12)

    C = np.cov(X_scaled, rowvar=False)
    if n_features == 1:
        C = np.array([[float(C)]])

    theta_max = max(_delta_theta_max(C, eigengap), 1e-12)
    theta_min = (2.0 / np.pi) * np.arcsin(
        np.clip(1.0 / (n_features * eigengap), 0.0, 1.0)
    )

    return X_scaled, base_vector, float(theta_max), float(theta_min)


def _map_uncached(X: pd.DataFrame, remaining_indices: list, n_jobs: int) -> float:
    X_scaled, base_vector, theta_max, theta_min = _shared_terms(X)

    delta = Parallel(n_jobs=n_jobs)(
        delayed(_delta_one)(p, X_scaled, base_vector)
        for p in remaining_indices
    )

    map_raw = float(np.mean(delta))
    return _scale_map(map_raw, theta_max, theta_min)


def _build_cache(X: pd.DataFrame, n_jobs: int) -> _MAPCache:
    X_scaled, base_vector, theta_max, theta_min = _shared_terms(X)

    delta = Parallel(n_jobs=n_jobs)(
        delayed(_delta_one)(p, X_scaled, base_vector)
        for p in range(X.shape[1])
    )

    return _MAPCache(
        columns=pd.Index(X.columns),
        delta=np.asarray(delta, dtype=float),
        theta_max=theta_max,
        theta_min=theta_min,
    )


def _get_cache(X: pd.DataFrame, n_jobs: int, force_rebuild: bool) -> _MAPCache:
    key = id(X)

    if not force_rebuild and key in _CACHE:
        ref, cache = _CACHE[key]
        if ref() is X:
            return cache

    cache = _build_cache(X, n_jobs=n_jobs)

    try:
        _CACHE[key] = (weakref.ref(X, lambda _: _drop_cache(key)), cache)
    except TypeError:
        _CACHE[key] = (lambda: X, cache)

    return cache


def _scale_map(map_raw: float, theta_max: float, theta_min: float) -> float:
    denom = max(theta_max - theta_min, 1e-12)
    return float(1 - ((map_raw - theta_min) / denom))


def MAP(
    X: pd.DataFrame,
    selected_features: list,
    n_jobs: int = -1,
    benchmark: bool = False,
    force_rebuild: bool = False,
) -> float:
    """
    Compute Mean Angle Perturbation (MAP).

    Parameters
    ----------
    X:
        Input data. A pandas DataFrame is recommended, especially when using
        string feature names or benchmark=True caching.
    selected_features:
        Selected feature names or integer column indices.
    n_jobs:
        Number of parallel jobs used by joblib.
    benchmark:
        If False, compute MAP directly for the not-selected features.
        If True, cache per-feature delta values for X and reuse them across calls.
    force_rebuild:
        If True and benchmark=True, rebuild the cache for X. Useful for multiple calculations on the same dataset.
    """
    X = _as_dataframe(X)
    n_features = X.shape[1]

    selected_indices = _selected_indices(selected_features, X.columns)
    remaining_indices = [i for i in range(n_features) if i not in selected_indices]

    if not remaining_indices:
        return 1

    if not benchmark:
        return _map_uncached(X, remaining_indices, n_jobs=n_jobs)

    cache = _get_cache(X, n_jobs=n_jobs, force_rebuild=force_rebuild)
    map_raw = float(np.mean(cache.delta[remaining_indices]))
    return _scale_map(map_raw, cache.theta_max, cache.theta_min)
