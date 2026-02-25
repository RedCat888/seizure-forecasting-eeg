"""
Handcrafted EEG features for baseline models.

Features computed per channel, then aggregated (mean + std) across channels:
- Bandpower (delta, theta, alpha, beta, gamma)
- Band ratios (theta/alpha, beta/alpha)
- Line length (spikiness measure)
- Spectral entropy
- Hjorth parameters (mobility, complexity)
- Statistical moments (kurtosis, variance)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal as sig
from scipy import stats
from omegaconf import DictConfig


# Default frequency bands
DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def extract_bandpower(
    x: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute bandpower for each frequency band using Welch's method.
    
    Args:
        x: Signal array [C, T] (channels x time)
        sfreq: Sampling frequency in Hz
        bands: Dict of band name to (low, high) frequency tuple
        
    Returns:
        Dict mapping band names to power arrays [C]
    """
    if bands is None:
        bands = DEFAULT_BANDS
    
    n_channels, n_times = x.shape
    nperseg = min(int(sfreq * 2), n_times)  # 2-second windows or less
    
    # Compute PSD for all channels
    freqs, psd = sig.welch(x, fs=sfreq, nperseg=nperseg, axis=1)
    
    # Extract power per band
    bandpower = {}
    for band_name, (low, high) in bands.items():
        # Find frequency indices
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        # Integrate power in band (simple sum)
        bp = np.sum(psd[:, idx], axis=1)
        bandpower[band_name] = bp
    
    return bandpower


def extract_band_ratios(
    bandpower: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute ratios between frequency bands.
    
    Args:
        bandpower: Dict from extract_bandpower
        
    Returns:
        Dict of ratio names to values [C]
    """
    ratios = {}
    
    # Theta/Alpha ratio (often elevated before seizures)
    if "theta" in bandpower and "alpha" in bandpower:
        alpha = np.maximum(bandpower["alpha"], 1e-10)  # Avoid division by zero
        ratios["theta_alpha"] = bandpower["theta"] / alpha
    
    # Beta/Alpha ratio
    if "beta" in bandpower and "alpha" in bandpower:
        alpha = np.maximum(bandpower["alpha"], 1e-10)
        ratios["beta_alpha"] = bandpower["beta"] / alpha
    
    # (Delta+Theta)/(Alpha+Beta) - slow/fast ratio
    if all(b in bandpower for b in ["delta", "theta", "alpha", "beta"]):
        slow = bandpower["delta"] + bandpower["theta"]
        fast = bandpower["alpha"] + bandpower["beta"]
        fast = np.maximum(fast, 1e-10)
        ratios["slow_fast"] = slow / fast
    
    return ratios


def extract_line_length(x: np.ndarray) -> np.ndarray:
    """
    Compute line length (sum of absolute first differences).
    
    Measures signal spikiness/complexity.
    
    Args:
        x: Signal array [C, T]
        
    Returns:
        Line length per channel [C]
    """
    diff = np.abs(np.diff(x, axis=1))
    return np.sum(diff, axis=1)


def extract_spectral_entropy(
    x: np.ndarray,
    sfreq: float,
    fmax: float = 45.0,
) -> np.ndarray:
    """
    Compute spectral entropy (normalized Shannon entropy of PSD).
    
    Low entropy = narrow-band, high entropy = broad-band.
    
    Args:
        x: Signal array [C, T]
        sfreq: Sampling frequency
        fmax: Maximum frequency to consider
        
    Returns:
        Spectral entropy per channel [C]
    """
    n_channels, n_times = x.shape
    nperseg = min(int(sfreq * 2), n_times)
    
    freqs, psd = sig.welch(x, fs=sfreq, nperseg=nperseg, axis=1)
    
    # Limit to fmax
    idx = freqs <= fmax
    psd = psd[:, idx]
    
    # Normalize to probability distribution
    psd_sum = np.sum(psd, axis=1, keepdims=True)
    psd_norm = psd / np.maximum(psd_sum, 1e-10)
    
    # Shannon entropy (with small epsilon to avoid log(0))
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=1)
    
    # Normalize by max possible entropy
    max_entropy = np.log(psd.shape[1])
    entropy_norm = entropy / max_entropy
    
    return entropy_norm


def extract_hjorth_params(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hjorth parameters (mobility and complexity).
    
    Mobility: sqrt(var(x') / var(x))
    Complexity: mobility(x') / mobility(x)
    
    Args:
        x: Signal array [C, T]
        
    Returns:
        Tuple of (mobility [C], complexity [C])
    """
    # First derivative
    dx = np.diff(x, axis=1)
    
    # Second derivative
    ddx = np.diff(dx, axis=1)
    
    # Variances
    var_x = np.var(x, axis=1)
    var_dx = np.var(dx, axis=1)
    var_ddx = np.var(ddx, axis=1)
    
    # Avoid division by zero
    var_x = np.maximum(var_x, 1e-10)
    var_dx = np.maximum(var_dx, 1e-10)
    
    # Mobility
    mobility = np.sqrt(var_dx / var_x)
    
    # Complexity
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / np.maximum(mobility, 1e-10)
    
    return mobility, complexity


def extract_statistical_features(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistical features per channel.
    
    Args:
        x: Signal array [C, T]
        
    Returns:
        Dict of feature names to values [C]
    """
    features = {}
    
    # Variance
    features["variance"] = np.var(x, axis=1)
    
    # Kurtosis (measure of tail heaviness)
    features["kurtosis"] = stats.kurtosis(x, axis=1)
    
    # Skewness
    features["skewness"] = stats.skew(x, axis=1)
    
    return features


def aggregate_channels(
    channel_features: np.ndarray,
) -> Tuple[float, float]:
    """
    Aggregate per-channel features to mean and std.
    
    Args:
        channel_features: Feature values per channel [C]
        
    Returns:
        Tuple of (mean, std)
    """
    return float(np.mean(channel_features)), float(np.std(channel_features))


def extract_features(
    x: np.ndarray,
    sfreq: float,
    cfg: Optional[DictConfig] = None,
) -> np.ndarray:
    """
    Extract all handcrafted features from an EEG window.
    
    Args:
        x: Signal array [C, T] (channels x time)
        sfreq: Sampling frequency in Hz
        cfg: Optional config with feature settings
        
    Returns:
        Feature vector [F] where F is the number of features
    """
    # Get settings from config or use defaults
    if cfg is not None and hasattr(cfg, "features"):
        bands = {k: tuple(v) for k, v in cfg.features.bands.items()}
        compute_bandpower = cfg.features.get("compute_bandpower", True)
        compute_ratios = cfg.features.get("compute_ratios", True)
        compute_ll = cfg.features.get("compute_line_length", True)
        compute_entropy = cfg.features.get("compute_spectral_entropy", True)
        compute_hjorth = cfg.features.get("compute_hjorth", True)
        compute_stats = cfg.features.get("compute_kurtosis", True)
    else:
        bands = DEFAULT_BANDS
        compute_bandpower = True
        compute_ratios = True
        compute_ll = True
        compute_entropy = True
        compute_hjorth = True
        compute_stats = True
    
    features = []
    
    # Bandpower features
    if compute_bandpower:
        bp = extract_bandpower(x, sfreq, bands)
        for band_name in sorted(bp.keys()):
            mean, std = aggregate_channels(bp[band_name])
            features.extend([mean, std])
    
    # Band ratios
    if compute_ratios and compute_bandpower:
        bp = extract_bandpower(x, sfreq, bands)
        ratios = extract_band_ratios(bp)
        for ratio_name in sorted(ratios.keys()):
            mean, std = aggregate_channels(ratios[ratio_name])
            features.extend([mean, std])
    
    # Line length
    if compute_ll:
        ll = extract_line_length(x)
        mean, std = aggregate_channels(ll)
        features.extend([mean, std])
    
    # Spectral entropy
    if compute_entropy:
        se = extract_spectral_entropy(x, sfreq)
        mean, std = aggregate_channels(se)
        features.extend([mean, std])
    
    # Hjorth parameters
    if compute_hjorth:
        mobility, complexity = extract_hjorth_params(x)
        mean_m, std_m = aggregate_channels(mobility)
        mean_c, std_c = aggregate_channels(complexity)
        features.extend([mean_m, std_m, mean_c, std_c])
    
    # Statistical features
    if compute_stats:
        stat_feats = extract_statistical_features(x)
        for feat_name in sorted(stat_feats.keys()):
            mean, std = aggregate_channels(stat_feats[feat_name])
            features.extend([mean, std])
    
    return np.array(features, dtype=np.float32)


def get_feature_names(cfg: Optional[DictConfig] = None) -> List[str]:
    """
    Get names of all features in the order they are extracted.
    
    Args:
        cfg: Optional config with feature settings
        
    Returns:
        List of feature names
    """
    # Get settings from config or use defaults
    if cfg is not None and hasattr(cfg, "features"):
        bands = list(cfg.features.bands.keys())
        compute_bandpower = cfg.features.get("compute_bandpower", True)
        compute_ratios = cfg.features.get("compute_ratios", True)
        compute_ll = cfg.features.get("compute_line_length", True)
        compute_entropy = cfg.features.get("compute_spectral_entropy", True)
        compute_hjorth = cfg.features.get("compute_hjorth", True)
        compute_stats = cfg.features.get("compute_kurtosis", True)
    else:
        bands = list(DEFAULT_BANDS.keys())
        compute_bandpower = True
        compute_ratios = True
        compute_ll = True
        compute_entropy = True
        compute_hjorth = True
        compute_stats = True
    
    names = []
    
    if compute_bandpower:
        for band in sorted(bands):
            names.extend([f"bp_{band}_mean", f"bp_{band}_std"])
    
    if compute_ratios and compute_bandpower:
        ratio_names = ["theta_alpha", "beta_alpha", "slow_fast"]
        for ratio in sorted(ratio_names):
            names.extend([f"ratio_{ratio}_mean", f"ratio_{ratio}_std"])
    
    if compute_ll:
        names.extend(["line_length_mean", "line_length_std"])
    
    if compute_entropy:
        names.extend(["spectral_entropy_mean", "spectral_entropy_std"])
    
    if compute_hjorth:
        names.extend([
            "hjorth_mobility_mean", "hjorth_mobility_std",
            "hjorth_complexity_mean", "hjorth_complexity_std",
        ])
    
    if compute_stats:
        stat_names = ["kurtosis", "skewness", "variance"]
        for stat in sorted(stat_names):
            names.extend([f"{stat}_mean", f"{stat}_std"])
    
    return names
