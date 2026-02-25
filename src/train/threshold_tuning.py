"""
Threshold tuning module for FAH-targeted alarm optimization.

Provides systematic threshold selection to achieve target False Alarm Rate per Hour (FAH)
while maximizing sensitivity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ThresholdResult:
    """Result of threshold tuning for a specific FAH target."""
    fah_target: float
    threshold: float
    achieved_fah: float
    sensitivity: float
    mean_warning_time_sec: float
    n_seizures: int
    n_detected: int
    n_false_alarms: int
    total_hours: float


@dataclass 
class AlarmEvent:
    """Represents a single alarm event."""
    time_sec: float
    risk_score: float
    is_true_positive: bool
    warning_time_sec: Optional[float] = None  # Time before seizure onset


def compute_alarm_metrics_at_threshold(
    times: np.ndarray,
    risk_scores: np.ndarray,
    seizure_intervals: List[Tuple[float, float]],
    threshold: float,
    refractory_sec: float = 1200.0,  # 20 minutes
    preictal_window_sec: float = 600.0,  # 10 minutes - alarm within this = true positive
    recording_hours: Optional[float] = None,
) -> Dict:
    """
    Compute alarm-level metrics at a specific threshold.
    
    Args:
        times: Window timestamps in seconds [N]
        risk_scores: Predicted risk scores [N]
        seizure_intervals: List of (onset, offset) tuples in seconds
        threshold: Classification threshold
        refractory_sec: Minimum time between alarms
        preictal_window_sec: Window before seizure where alarm counts as TP
        recording_hours: Total recording duration in hours (for FAH)
        
    Returns:
        Dict with FAH, sensitivity, warning times, etc.
    """
    if len(times) == 0 or len(risk_scores) == 0:
        return {
            "fah": 0.0,
            "sensitivity": 0.0,
            "mean_warning_time": 0.0,
            "n_alarms": 0,
            "n_true_positives": 0,
            "n_false_alarms": 0,
            "n_seizures": len(seizure_intervals),
            "warning_times": [],
        }
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    risk_scores = risk_scores[sort_idx]
    
    # Find alarm times (threshold crossings with refractory)
    above_threshold = risk_scores >= threshold
    alarm_times = []
    last_alarm_time = -np.inf
    
    for i, (t, above) in enumerate(zip(times, above_threshold)):
        if above and (t - last_alarm_time) >= refractory_sec:
            alarm_times.append(t)
            last_alarm_time = t
    
    alarm_times = np.array(alarm_times)
    
    # Classify alarms as TP or FP
    seizure_detected = [False] * len(seizure_intervals)
    warning_times = []
    true_positives = 0
    false_alarms = 0
    
    for alarm_t in alarm_times:
        is_tp = False
        for i, (onset, offset) in enumerate(seizure_intervals):
            # Alarm is TP if it's within preictal_window_sec before onset
            # and the seizure hasn't been detected yet
            if (onset - preictal_window_sec) <= alarm_t < onset and not seizure_detected[i]:
                is_tp = True
                seizure_detected[i] = True
                warning_times.append(onset - alarm_t)
                break
        
        if is_tp:
            true_positives += 1
        else:
            false_alarms += 1
    
    # Compute recording hours if not provided
    if recording_hours is None:
        recording_hours = (times[-1] - times[0]) / 3600.0
        recording_hours = max(recording_hours, 0.1)  # Avoid division by zero
    
    # Compute metrics
    fah = false_alarms / recording_hours if recording_hours > 0 else 0.0
    sensitivity = sum(seizure_detected) / len(seizure_intervals) if seizure_intervals else 0.0
    mean_warning = np.mean(warning_times) if warning_times else 0.0
    
    return {
        "fah": fah,
        "sensitivity": sensitivity,
        "mean_warning_time": mean_warning,
        "n_alarms": len(alarm_times),
        "n_true_positives": true_positives,
        "n_false_alarms": false_alarms,
        "n_seizures": len(seizure_intervals),
        "n_detected": sum(seizure_detected),
        "warning_times": warning_times,
        "alarm_times": alarm_times.tolist() if len(alarm_times) > 0 else [],
    }


def tune_threshold_for_fah(
    times: np.ndarray,
    risk_scores: np.ndarray,
    seizure_intervals: List[Tuple[float, float]],
    target_fah: float,
    n_thresholds: int = 100,
    refractory_sec: float = 1200.0,
    preictal_window_sec: float = 600.0,
    recording_hours: Optional[float] = None,
) -> ThresholdResult:
    """
    Find threshold that achieves target FAH (without exceeding it).
    
    Args:
        times: Window timestamps
        risk_scores: Predicted risk scores
        seizure_intervals: List of (onset, offset) tuples
        target_fah: Target false alarm rate per hour
        n_thresholds: Number of thresholds to try
        refractory_sec: Refractory period between alarms
        preictal_window_sec: Window for true positive classification
        recording_hours: Total recording hours
        
    Returns:
        ThresholdResult with optimal threshold and metrics
    """
    # Try thresholds from low to high
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    best_result = None
    best_sensitivity = -1.0
    
    for thresh in thresholds:
        metrics = compute_alarm_metrics_at_threshold(
            times, risk_scores, seizure_intervals, thresh,
            refractory_sec, preictal_window_sec, recording_hours
        )
        
        # We want FAH <= target and maximum sensitivity
        if metrics["fah"] <= target_fah:
            if metrics["sensitivity"] > best_sensitivity:
                best_sensitivity = metrics["sensitivity"]
                best_result = ThresholdResult(
                    fah_target=target_fah,
                    threshold=thresh,
                    achieved_fah=metrics["fah"],
                    sensitivity=metrics["sensitivity"],
                    mean_warning_time_sec=metrics["mean_warning_time"],
                    n_seizures=metrics["n_seizures"],
                    n_detected=metrics["n_detected"],
                    n_false_alarms=metrics["n_false_alarms"],
                    total_hours=recording_hours or (times[-1] - times[0]) / 3600.0,
                )
    
    # If no threshold achieves target FAH, use highest threshold
    if best_result is None:
        thresh = thresholds[-1]
        metrics = compute_alarm_metrics_at_threshold(
            times, risk_scores, seizure_intervals, thresh,
            refractory_sec, preictal_window_sec, recording_hours
        )
        best_result = ThresholdResult(
            fah_target=target_fah,
            threshold=thresh,
            achieved_fah=metrics["fah"],
            sensitivity=metrics["sensitivity"],
            mean_warning_time_sec=metrics["mean_warning_time"],
            n_seizures=metrics["n_seizures"],
            n_detected=metrics["n_detected"],
            n_false_alarms=metrics["n_false_alarms"],
            total_hours=recording_hours or (times[-1] - times[0]) / 3600.0,
        )
    
    return best_result


def tune_thresholds_for_multiple_targets(
    times: np.ndarray,
    risk_scores: np.ndarray,
    seizure_intervals: List[Tuple[float, float]],
    fah_targets: List[float] = [0.1, 0.2, 0.5, 1.0],
    **kwargs,
) -> Dict[float, ThresholdResult]:
    """
    Tune thresholds for multiple FAH targets.
    
    Returns:
        Dict mapping FAH target to ThresholdResult
    """
    results = {}
    for target in fah_targets:
        results[target] = tune_threshold_for_fah(
            times, risk_scores, seizure_intervals, target, **kwargs
        )
    return results


def compute_threshold_curve(
    times: np.ndarray,
    risk_scores: np.ndarray,
    seizure_intervals: List[Tuple[float, float]],
    n_thresholds: int = 50,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Compute FAH and sensitivity curves across thresholds.
    
    Returns:
        Dict with 'thresholds', 'fah', 'sensitivity', 'warning_time' arrays
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    fah_values = []
    sensitivity_values = []
    warning_times = []
    
    for thresh in thresholds:
        metrics = compute_alarm_metrics_at_threshold(
            times, risk_scores, seizure_intervals, thresh, **kwargs
        )
        fah_values.append(metrics["fah"])
        sensitivity_values.append(metrics["sensitivity"])
        warning_times.append(metrics["mean_warning_time"])
    
    return {
        "thresholds": thresholds,
        "fah": np.array(fah_values),
        "sensitivity": np.array(sensitivity_values),
        "warning_time": np.array(warning_times),
    }


def apply_risk_smoothing(
    risk_scores: np.ndarray,
    method: str = "ema",
    alpha: float = 0.2,
    window_size: int = 6,
) -> np.ndarray:
    """
    Apply temporal smoothing to risk scores.
    
    Args:
        risk_scores: Raw risk scores [N]
        method: "ema" (exponential moving average) or "moving_avg"
        alpha: EMA decay factor (smaller = more smoothing)
        window_size: Window size for moving average
        
    Returns:
        Smoothed risk scores
    """
    if len(risk_scores) == 0:
        return risk_scores
    
    smoothed = np.zeros_like(risk_scores)
    
    if method == "ema":
        smoothed[0] = risk_scores[0]
        for i in range(1, len(risk_scores)):
            smoothed[i] = alpha * risk_scores[i] + (1 - alpha) * smoothed[i-1]
    
    elif method == "moving_avg":
        for i in range(len(risk_scores)):
            start = max(0, i - window_size + 1)
            smoothed[i] = np.mean(risk_scores[start:i+1])
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed


def apply_persistence_filter(
    risk_scores: np.ndarray,
    threshold: float,
    k_consecutive: int = 3,
) -> np.ndarray:
    """
    Apply persistence filter - require K consecutive windows above threshold.
    
    Args:
        risk_scores: Risk scores [N]
        threshold: Activation threshold
        k_consecutive: Number of consecutive windows required
        
    Returns:
        Binary alarm signal (1 = alarm, 0 = no alarm)
    """
    above = (risk_scores >= threshold).astype(int)
    alarms = np.zeros_like(above)
    
    consecutive_count = 0
    for i in range(len(above)):
        if above[i]:
            consecutive_count += 1
            if consecutive_count >= k_consecutive:
                alarms[i] = 1
        else:
            consecutive_count = 0
    
    return alarms


def apply_hysteresis(
    risk_scores: np.ndarray,
    trigger_threshold: float,
    reset_threshold: float,
) -> np.ndarray:
    """
    Apply hysteresis to reduce alarm chatter.
    
    Args:
        risk_scores: Risk scores [N]
        trigger_threshold: Threshold to trigger alarm
        reset_threshold: Threshold to reset (must be lower than trigger)
        
    Returns:
        Binary alarm signal with hysteresis
    """
    if reset_threshold >= trigger_threshold:
        reset_threshold = trigger_threshold - 0.1
    
    alarms = np.zeros(len(risk_scores), dtype=int)
    in_alarm_state = False
    
    for i, score in enumerate(risk_scores):
        if not in_alarm_state and score >= trigger_threshold:
            in_alarm_state = True
            alarms[i] = 1
        elif in_alarm_state and score < reset_threshold:
            in_alarm_state = False
        elif in_alarm_state:
            alarms[i] = 1  # Stay in alarm state
    
    return alarms


class AlarmProcessor:
    """
    Configurable alarm post-processor combining smoothing, persistence, and hysteresis.
    """
    
    def __init__(
        self,
        smoothing_method: Optional[str] = None,  # "ema" or "moving_avg"
        smoothing_alpha: float = 0.2,
        smoothing_window: int = 6,
        persistence_k: int = 1,  # 1 = no persistence
        use_hysteresis: bool = False,
        hysteresis_gap: float = 0.1,  # Reset = trigger - gap
        refractory_sec: float = 1200.0,
    ):
        self.smoothing_method = smoothing_method
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_window = smoothing_window
        self.persistence_k = persistence_k
        self.use_hysteresis = use_hysteresis
        self.hysteresis_gap = hysteresis_gap
        self.refractory_sec = refractory_sec
    
    def process(
        self,
        times: np.ndarray,
        risk_scores: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process risk scores and return alarm times.
        
        Returns:
            Tuple of (processed_risk_scores, alarm_mask)
        """
        # Step 1: Smoothing
        if self.smoothing_method:
            risk_scores = apply_risk_smoothing(
                risk_scores,
                method=self.smoothing_method,
                alpha=self.smoothing_alpha,
                window_size=self.smoothing_window,
            )
        
        # Step 2: Persistence filter
        if self.persistence_k > 1:
            alarm_mask = apply_persistence_filter(
                risk_scores, threshold, self.persistence_k
            )
        elif self.use_hysteresis:
            alarm_mask = apply_hysteresis(
                risk_scores, threshold, threshold - self.hysteresis_gap
            )
        else:
            alarm_mask = (risk_scores >= threshold).astype(int)
        
        # Step 3: Apply refractory period
        alarm_times = []
        last_alarm = -np.inf
        final_mask = np.zeros_like(alarm_mask)
        
        for i, (t, is_alarm) in enumerate(zip(times, alarm_mask)):
            if is_alarm and (t - last_alarm) >= self.refractory_sec:
                alarm_times.append(t)
                last_alarm = t
                final_mask[i] = 1
        
        return risk_scores, final_mask
    
    def get_config_str(self) -> str:
        """Return string describing this configuration."""
        parts = []
        if self.smoothing_method:
            if self.smoothing_method == "ema":
                parts.append(f"EMA(α={self.smoothing_alpha})")
            else:
                parts.append(f"MA(n={self.smoothing_window})")
        if self.persistence_k > 1:
            parts.append(f"Persist(k={self.persistence_k})")
        if self.use_hysteresis:
            parts.append(f"Hyst(gap={self.hysteresis_gap})")
        return "+".join(parts) if parts else "baseline"
