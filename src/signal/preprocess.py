"""
EEG preprocessing pipeline using MNE.

Steps:
1. Load EDF file
2. Drop non-EEG channels
3. Standardize channel order to CANONICAL_CHANNELS
4. Resample to target frequency
5. Bandpass filter (0.5-45 Hz)
6. Notch filter (60 Hz line noise)
7. Re-reference to common average
8. Artifact detection (amplitude threshold)

CHANNEL CONSISTENCY:
- Uses CANONICAL_CHANNELS list to ensure identical channel order
- Logs any missing/dropped channels
- Asserts output shape is constant
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import mne
from omegaconf import DictConfig


# Standard CHB-MIT channel set (most common across subjects)
# These are bipolar montage channels
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

# CANONICAL CHANNELS - strictly enforced order across ALL samples
# This is the intersection of channels present in most CHB-MIT subjects
# Any file missing these channels will have zeros filled in
CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

# Channel name mapping for normalization (handle variations)
CHANNEL_ALIASES = {
    "T8-P8-0": "T8-P8",
    "T8-P8-1": "T8-P8",
    "FP1-F7-0": "FP1-F7",
    "P8-O2-0": "P8-O2",
}


def is_eeg_channel(ch_name: str) -> bool:
    """
    Check if a channel name looks like an EEG channel.
    
    EEG channels in CHB-MIT are typically bipolar like "FP1-F7" or "F7-T7".
    """
    ch_upper = ch_name.upper()
    
    # Skip known non-EEG patterns
    skip_patterns = [
        "ECG", "EKG", "EMG", "EOG", "LOC", "ROC",
        "VNS", "RESP", "PHOTIC", "."  # Removed "-" which was incorrectly listed
    ]
    
    for pat in skip_patterns:
        if pat in ch_upper:
            return False
    
    # Check for hyphenated bipolar montage pattern (standard for CHB-MIT)
    if "-" in ch_name:
        parts = ch_name.split("-")
        # Should have at least 2 parts for bipolar montage
        if len(parts) >= 2:
            # First part should be electrode name (letters + optional digit)
            # e.g., FP1, F7, T7, P7, O1, FZ, CZ, PZ
            first = parts[0].upper()
            second = parts[1].upper()
            
            # Valid electrode patterns
            import re
            electrode_pattern = r'^[A-Z]+[0-9]?$'  # Letters followed by optional digit
            
            # Check both parts (handle MNE renaming like T8-P8-0)
            first_valid = re.match(electrode_pattern, first) is not None
            second_base = second.rstrip("0123456789")  # Handle T8-P8-0 -> P8
            if second_base.endswith("-"):
                second_base = second_base[:-1]
            second_valid = re.match(electrode_pattern, second_base) is not None if second_base else True
            
            return first_valid and second_valid
    
    return False


def get_eeg_channels(raw: mne.io.Raw) -> List[str]:
    """
    Get list of EEG channels from raw object.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        List of EEG channel names
    """
    eeg_channels = []
    for ch in raw.ch_names:
        if is_eeg_channel(ch):
            eeg_channels.append(ch)
    return eeg_channels


def normalize_channel_name(ch_name: str) -> str:
    """
    Normalize channel name to canonical form (uppercase).
    
    Handles MNE duplicate renaming (e.g., T8-P8-0 -> T8-P8).
    """
    ch_upper = ch_name.upper()
    
    # Check aliases first
    for alias, canonical in CHANNEL_ALIASES.items():
        if ch_upper == alias.upper():
            return canonical.upper()
    
    # Strip trailing numbers from duplicates (e.g., T8-P8-0 -> T8-P8)
    # Pattern: ends with -N where N is one or more digits
    import re
    match = re.match(r'^(.+)-\d+$', ch_upper)
    if match:
        return match.group(1)
    
    return ch_upper


def get_common_channels(raw: mne.io.Raw) -> List[str]:
    """
    Get intersection of raw channels with standard channels.
    Falls back to all EEG-like channels if no matches.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        List of channels to use
    """
    raw_channels_upper = {ch.upper(): ch for ch in raw.ch_names}
    
    # Try to match standard channels (handling duplicates like T8-P8-0)
    common = []
    for std_ch in STANDARD_CHANNELS:
        std_upper = std_ch.upper()
        
        # Check for exact match
        if std_upper in raw_channels_upper:
            common.append(raw_channels_upper[std_upper])
        else:
            # Check for renamed duplicates (e.g., T8-P8-0, T8-P8-1)
            for raw_upper, raw_orig in raw_channels_upper.items():
                if raw_upper.startswith(std_upper + "-") or raw_upper.rstrip("0123456789-") == std_upper:
                    if raw_orig not in common:
                        common.append(raw_orig)
                        break
    
    # If we got enough standard channels, use them
    if len(common) >= 10:
        return common
    
    # Fallback: use all EEG-like channels
    return get_eeg_channels(raw)


def enforce_canonical_channels(
    data: np.ndarray,
    channel_names: List[str],
    canonical: List[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Enforce canonical channel ordering across all samples.
    
    Args:
        data: EEG data array [C, T]
        channel_names: Current channel names
        canonical: Target canonical channel list (defaults to CANONICAL_CHANNELS)
        verbose: Whether to print channel mapping info
        
    Returns:
        Tuple of (reordered data [C', T], canonical channel names, info dict)
    """
    if canonical is None:
        canonical = CANONICAL_CHANNELS
    
    n_times = data.shape[1]
    n_canonical = len(canonical)
    
    # Create output array (zeros for missing channels)
    output = np.zeros((n_canonical, n_times), dtype=data.dtype)
    
    # Build normalized name -> index mapping for input channels
    input_map = {}
    for i, ch in enumerate(channel_names):
        norm_name = normalize_channel_name(ch)
        input_map[norm_name] = i
    
    # Map channels
    found_channels = []
    missing_channels = []
    
    for i, canonical_ch in enumerate(canonical):
        canonical_upper = canonical_ch.upper()
        
        if canonical_upper in input_map:
            src_idx = input_map[canonical_upper]
            output[i] = data[src_idx]
            found_channels.append(canonical_ch)
        else:
            missing_channels.append(canonical_ch)
            # Output remains zeros
    
    info = {
        "n_found": len(found_channels),
        "n_missing": len(missing_channels),
        "found_channels": found_channels,
        "missing_channels": missing_channels,
        "original_channels": channel_names,
    }
    
    if verbose and missing_channels:
        print(f"  [CHANNEL] Missing {len(missing_channels)} channels: {missing_channels[:5]}...")
    
    return output, list(canonical), info


def log_channel_info(edf_path: Path, channel_names: List[str], log_file: Path = None):
    """
    Log channel information for an EDF file.
    
    Args:
        edf_path: Path to EDF file
        channel_names: List of channels found
        log_file: Optional path to log file
    """
    info_line = f"{edf_path.name}: {len(channel_names)} channels - {', '.join(channel_names[:5])}..."
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{edf_path.name},{len(channel_names)},{','.join(channel_names)}\n")
    
    return info_line


def load_edf_raw(
    edf_path: str | Path,
    verbose: bool = False,
) -> mne.io.Raw:
    """
    Load an EDF file with MNE.
    
    Args:
        edf_path: Path to EDF file
        verbose: Whether to print MNE messages
        
    Returns:
        MNE Raw object
    """
    mne.set_log_level("WARNING" if not verbose else "INFO")
    
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=True,
        verbose=verbose,
    )
    
    return raw


def preprocess_edf(
    edf_path: str | Path,
    cfg: Optional[DictConfig] = None,
    target_sfreq: float = 256.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 45.0,
    notch_freq: float = 60.0,
    notch_width: float = 2.0,
    verbose: bool = False,
    enforce_canonical: bool = True,
    channel_log_file: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str], float, Optional[Dict]]:
    """
    Preprocess an entire EDF file with canonical channel ordering.
    
    Args:
        edf_path: Path to EDF file
        cfg: Optional config (overrides other params)
        target_sfreq: Target sampling frequency
        bandpass_low: Highpass cutoff
        bandpass_high: Lowpass cutoff
        notch_freq: Notch filter center frequency
        notch_width: Notch filter bandwidth
        verbose: Whether to print messages
        enforce_canonical: Whether to enforce CANONICAL_CHANNELS order
        channel_log_file: Optional file to log channel info
        
    Returns:
        Tuple of (data array [C, T], channel names, sfreq, channel_info)
        If enforce_canonical=True, channels are ALWAYS in CANONICAL_CHANNELS order
    """
    edf_path = Path(edf_path)
    
    # Override with config if provided
    if cfg is not None:
        target_sfreq = cfg.signal.get("target_sfreq", target_sfreq)
        bandpass_low = cfg.signal.get("bandpass_low", bandpass_low)
        bandpass_high = cfg.signal.get("bandpass_high", bandpass_high)
        notch_freq = cfg.signal.get("notch_freq", notch_freq)
        notch_width = cfg.signal.get("notch_width", notch_width)
    
    mne.set_log_level("WARNING" if not verbose else "INFO")
    
    # Load raw data
    raw = load_edf_raw(edf_path, verbose=verbose)
    
    # Get EEG channels
    eeg_channels = get_eeg_channels(raw)
    
    if len(eeg_channels) < 5:
        raise ValueError(f"Not enough EEG channels found: {len(eeg_channels)}")
    
    # Log original channels
    if channel_log_file:
        log_channel_info(edf_path, eeg_channels, channel_log_file)
    
    # Pick only EEG channels
    raw.pick(eeg_channels)
    
    # Resample if needed
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)
    
    # Bandpass filter
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method="fir",
        fir_design="firwin",
        verbose=verbose,
    )
    
    # Notch filter
    raw.notch_filter(
        freqs=notch_freq,
        notch_widths=notch_width,
        method="fir",
        fir_design="firwin",
        verbose=verbose,
    )
    
    # Re-reference to common average
    raw.set_eeg_reference("average", projection=False, verbose=verbose)
    
    # Get data as numpy array
    data = raw.get_data()  # [n_channels, n_times]
    channels = raw.ch_names
    sfreq = raw.info["sfreq"]
    
    # Enforce canonical channel ordering
    channel_info = None
    if enforce_canonical:
        data, channels, channel_info = enforce_canonical_channels(
            data, channels, CANONICAL_CHANNELS, verbose=verbose
        )
        
        # Assert output shape is as expected
        assert data.shape[0] == len(CANONICAL_CHANNELS), \
            f"Expected {len(CANONICAL_CHANNELS)} channels, got {data.shape[0]}"
    
    return data, channels, sfreq, channel_info


def preprocess_window(
    window: np.ndarray,
    sfreq: float,
    amp_uv_thresh: float = 500.0,
) -> Tuple[np.ndarray, bool]:
    """
    Check and optionally preprocess a single window.
    
    Args:
        window: Window data [C, T]
        sfreq: Sampling frequency
        amp_uv_thresh: Amplitude threshold in microvolts
        
    Returns:
        Tuple of (processed window, is_valid)
    """
    # Convert to microvolts if needed (MNE uses volts)
    window_uv = window * 1e6
    
    # Check amplitude threshold
    peak_to_peak = np.ptp(window_uv, axis=1)  # Per channel
    max_ptp = np.max(peak_to_peak)
    
    is_valid = max_ptp <= amp_uv_thresh
    
    return window, is_valid


def check_window_artifact(
    window: np.ndarray,
    amp_uv_thresh: float = 500.0,
) -> bool:
    """
    Check if window contains artifacts.
    
    Args:
        window: Window data [C, T] in volts
        amp_uv_thresh: Threshold in microvolts
        
    Returns:
        True if window is clean (no artifact)
    """
    # Convert to microvolts
    window_uv = window * 1e6
    
    # Check peak-to-peak
    max_ptp = np.max(np.ptp(window_uv, axis=1))
    
    return max_ptp <= amp_uv_thresh


def create_preprocessing_plots(
    edf_path: str | Path,
    output_dir: str | Path,
    cfg: Optional[DictConfig] = None,
) -> Dict[str, Path]:
    """
    Create before/after preprocessing visualization plots.
    
    Args:
        edf_path: Path to EDF file
        output_dir: Directory to save plots
        cfg: Optional config
        
    Returns:
        Dict of plot names to paths
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mne.set_log_level("WARNING")
    
    # Load raw (before preprocessing)
    raw_before = load_edf_raw(edf_path)
    
    # Get EEG channels only for before
    eeg_ch_before = get_eeg_channels(raw_before)
    raw_before.pick(eeg_ch_before[:8])  # First 8 EEG channels
    
    # Get preprocessed
    data_after, channels, sfreq = preprocess_edf(edf_path, cfg=cfg)
    
    # Select a 10-second segment for visualization
    start_sec = 60  # Start at 1 minute
    duration_sec = 10
    n_samples = int(duration_sec * sfreq)
    start_sample = int(start_sec * raw_before.info["sfreq"])
    
    plots = {}
    
    # Plot 1: Raw vs preprocessed EEG
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Before
    n_ch_show = min(8, len(raw_before.ch_names))
    data_before = raw_before.get_data()[:n_ch_show, start_sample:start_sample+n_samples]
    times = np.arange(data_before.shape[1]) / raw_before.info["sfreq"]
    
    for i in range(n_ch_show):
        offset = i * 100e-6  # Offset for visibility
        axes[0].plot(times, data_before[i] + offset, linewidth=0.5)
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].set_title("Raw EEG (before preprocessing)")
    axes[0].set_xlim([0, duration_sec])
    
    # After
    start_sample_after = int(start_sec * sfreq)
    n_ch_show = min(8, len(channels))
    if start_sample_after + n_samples <= data_after.shape[1]:
        data_after_seg = data_after[:n_ch_show, start_sample_after:start_sample_after+n_samples]
    else:
        data_after_seg = data_after[:n_ch_show, :n_samples]
    times_after = np.arange(data_after_seg.shape[1]) / sfreq
    
    for i in range(n_ch_show):
        offset = i * 100e-6
        axes[1].plot(times_after, data_after_seg[i] + offset, linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].set_title("Preprocessed EEG (filtered, re-referenced)")
    
    plt.tight_layout()
    eeg_plot_path = output_dir / "raw_vs_preprocessed_eeg.png"
    plt.savefig(eeg_plot_path, dpi=150)
    plt.close()
    plots["eeg_comparison"] = eeg_plot_path
    
    # Plot 2: PSD before and after
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSD before (manual computation)
    from scipy import signal as sig
    freqs_b, psd_b = sig.welch(data_before[0], fs=raw_before.info["sfreq"], nperseg=int(raw_before.info["sfreq"]*2))
    axes[0].semilogy(freqs_b, psd_b)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (V²/Hz)")
    axes[0].set_title("PSD - Before Preprocessing")
    axes[0].set_xlim([0, 100])
    
    # PSD after
    freqs, psd = sig.welch(data_after[0], fs=sfreq, nperseg=int(sfreq*2))
    axes[1].semilogy(freqs, psd)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (V²/Hz)")
    axes[1].set_title("PSD - After Preprocessing")
    axes[1].set_xlim([0, 50])
    
    plt.tight_layout()
    psd_plot_path = output_dir / "psd_before_after.png"
    plt.savefig(psd_plot_path, dpi=150)
    plt.close()
    plots["psd_comparison"] = psd_plot_path
    
    return plots
