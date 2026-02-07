import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from tslearn.metrics import dtw

def _extract_feature(real: np.ndarray, gen: np.ndarray, feature_idx: int):

    if real.ndim != 3 or gen.ndim != 3:
        raise ValueError("Input data must be 3D array, shape (n_samples, T, n_features)")
    if real.shape[1] != gen.shape[1]:
        raise ValueError("Real data and generated data must have the same time steps T")
    return real[:, :, feature_idx], gen[:, :, feature_idx]

def rmse_mean(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> float:

    r, g = _extract_feature(real, gen, feature_idx)
    mu_r = r.mean(axis=0)
    mu_g = g.mean(axis=0)
    return np.sqrt(np.mean((mu_r - mu_g)**2))

def mape_mean(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> float:

    r, g = _extract_feature(real, gen, feature_idx)
    mu_r = r.mean(axis=0)
    mu_g = g.mean(axis=0)
    return np.mean(np.abs((mu_r - mu_g) / (mu_r + 1e-10))) * 100

def dtw_distance_mean(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> float:

    r, g = _extract_feature(real, gen, feature_idx)
    mu_r = r.mean(axis=0)
    mu_g = g.mean(axis=0)
    return float(dtw(mu_r, mu_g))

def wasserstein_distance_mean(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> float:

    r, g = _extract_feature(real, gen, feature_idx)
    T = r.shape[1]
    dists = [wasserstein_distance(r[:, t], g[:, t]) for t in range(T)]
    return float(np.mean(dists))

def ks_statistic_series(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> np.ndarray:

    r, g = _extract_feature(real, gen, feature_idx)
    T = r.shape[1]
    return np.array([ks_2samp(r[:, t], g[:, t])[0] for t in range(T)])

def ks_pvalue_series(real: np.ndarray, gen: np.ndarray, feature_idx: int) -> np.ndarray:

    r, g = _extract_feature(real, gen, feature_idx)
    T = r.shape[1]
    return np.array([ks_2samp(r[:, t], g[:, t])[1] for t in range(T)])