import weakref
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class _MAPCache:
    columns: pd.Index
    delta: np.ndarray
    theta_max: float


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


def _delta_theta_max(lam: np.ndarray, Vt: np.ndarray, eigengap: float) -> float:
    W = Vt**2
    cpp = lam @ W
    off_norm2 = np.clip((lam**2) @ W - cpp**2, 0.0, None)
    dk_norm = (cpp + np.sqrt(cpp**2 + 4.0 * off_norm2)) / 2.0

    sin_max = np.clip(dk_norm.max() / eigengap, 0.0, 1.0)
    return float((2.0 / np.pi) * np.arcsin(sin_max))


def _delta_direct(p: int, X_scaled: np.ndarray, base_vector: np.ndarray) -> float:
    X_comp = X_scaled.copy()
    X_comp[:, p] = 0.0

    comp_vector = np.linalg.svd(X_comp, full_matrices=False)[2][0]
    dot = np.clip(abs(np.dot(base_vector, comp_vector)), 0.0, 1.0)
    return float((2.0 / np.pi) * np.arccos(dot))


def _delta(indices, X_scaled: np.ndarray, lam: np.ndarray, Vt: np.ndarray,
           n_iter: int = 100, chunk_elems: int = 2**24) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    n_features = Vt.shape[1]
    l1, eigengap = lam[0], lam[0] - lam[1]
    D = l1 - lam
    if Vt.shape[0] < n_features:
        D = np.append(D, l1)

    delta = np.zeros(len(indices))
    fallback = []
    rows = max(1, chunk_elems // len(D))

    for start in range(0, len(indices), rows):
        block = np.arange(start, min(len(indices), start + rows))
        W = (Vt[:, indices[block]] ** 2).T
        if len(D) > W.shape[1]:
            W = np.hstack([W, np.clip(1.0 - W.sum(1, keepdims=True), 0.0, None)])

        live = W[:, 0] > 1e-28
        w1, Wr, Dr = W[live, 0], W[live, 1:], D[1:]

        def g(t):
            return w1 / t + (Wr / (t[:, None] - Dr)).sum(1)

        lo = np.full(len(w1), np.log(eigengap * 1e-30))
        hi = np.full(len(w1), np.log(eigengap * (1 - 1e-12)))
        ok = g(np.exp(hi)) < 0
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            pos = g(np.exp(mid)) > 0
            lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)

        t = np.exp((lo + hi) / 2)
        rest = (Wr / (t[:, None] - Dr) ** 2).sum(1)
        delta[block[live]] = (2.0 / np.pi) * np.arctan2(np.sqrt(rest), np.sqrt(w1) / t)
        fallback.extend(block[live][~ok].tolist())

    for i in fallback:
        delta[i] = _delta_direct(indices[i], X_scaled, Vt[0])
    return delta


def _shared_terms(X: pd.DataFrame):
    n_samples, n_features = X.shape

    X_scaled = StandardScaler().fit_transform(X.to_numpy(dtype=float))

    _, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    lam = s**2 / (n_samples - 1)
    eigengap = max(lam[0] - lam[1], 1e-12)

    theta_max = max(_delta_theta_max(lam, Vt, eigengap), 1e-12)

    return X_scaled, lam, Vt, float(theta_max)


def _map_uncached(X: pd.DataFrame, remaining_indices: list) -> float:
    X_scaled, lam, Vt, theta_max = _shared_terms(X)

    delta = _delta(remaining_indices, X_scaled, lam, Vt)

    map_raw = float(np.mean(delta))
    return _scale_map(map_raw, theta_max)


def _build_cache(X: pd.DataFrame) -> _MAPCache:
    X_scaled, lam, Vt, theta_max = _shared_terms(X)

    delta = _delta(range(X.shape[1]), X_scaled, lam, Vt)

    return _MAPCache(
        columns=pd.Index(X.columns),
        delta=np.asarray(delta, dtype=float),
        theta_max=theta_max,
    )


def _get_cache(X: pd.DataFrame, force_rebuild: bool) -> _MAPCache:
    key = id(X)

    if not force_rebuild and key in _CACHE:
        ref, cache = _CACHE[key]
        if ref() is X:
            return cache

    cache = _build_cache(X)

    try:
        _CACHE[key] = (weakref.ref(X, lambda _: _drop_cache(key)), cache)
    except TypeError:
        _CACHE[key] = (lambda: X, cache)

    return cache


def _scale_map(map_raw: float, theta_max: float) -> float:
    return float(1 - map_raw / max(theta_max, 1e-12))


def MAP(
    X: pd.DataFrame,
    selected_features: list,
    benchmark: bool = False,
    force_rebuild: bool = False,
) -> float:
    """
    Compute Mean Angular Preservation (MAP).

    Parameters
    ----------
    X:
        Input data. A pandas DataFrame is recommended, especially when using
        string feature names or benchmark=True caching.
    selected_features:
        Selected feature names or integer column indices.
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
        return _map_uncached(X, remaining_indices)

    cache = _get_cache(X, force_rebuild=force_rebuild)
    map_raw = float(np.mean(cache.delta[remaining_indices]))
    return _scale_map(map_raw, cache.theta_max)
