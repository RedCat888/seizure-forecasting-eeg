"""
Alarm-level evaluation for seizure forecasting.

Implements clinically meaningful metrics:
- False Alarm Rate (FAH): False alarms per hour
- Seizure Sensitivity: Fraction of seizures with timely warning
- Time-to-Warning: Lead time before seizure onset
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from omegaconf import DictConfig


@dataclass
class Alarm:
    """Represents a single alarm event."""
    time_sec: float
    risk_score: float
    is_true_alarm: bool  # True if followed by seizure within preictal window


@dataclass
class SeizureEvent:
    """Represents a seizure with alarm evaluation."""
    onset_sec: float
    offset_sec: float
    was_predicted: bool  # True if alarm occurred before onset
    first_alarm_time: Optional[float]  # Time of first warning alarm
    warning_time: Optional[float]  # Lead time (onset - first_alarm_time)


@dataclass
class AlarmMetrics:
    """Alarm-level evaluation metrics."""
    false_alarms_per_hour: float
    sensitivity: float  # Fraction of seizures predicted
    mean_warning_time: float  # Average lead time (seconds)
    n_seizures: int
    n_predicted: int
    n_false_alarms: int
    total_duration_hours: float


class AlarmEvaluator:
    """
    Evaluates seizure forecasting at the alarm level.
    
    Given risk scores over time, determines:
    - When alarms would trigger (risk > threshold)
    - Which seizures were successfully predicted
    - False alarm rate
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        refractory_min: float = 20.0,
        preictal_min: float = 10.0,
        gap_sec: float = 30.0,
    ):
        """
        Args:
            threshold: Risk threshold for alarm
            refractory_min: Minimum time between alarms (minutes)
            preictal_min: Preictal horizon (minutes)
            gap_sec: Gap before onset where alarms don't count
        """
        self.threshold = threshold
        self.refractory_sec = refractory_min * 60
        self.preictal_sec = preictal_min * 60
        self.gap_sec = gap_sec
    
    def evaluate_recording(
        self,
        risk_scores: np.ndarray,
        window_times: np.ndarray,
        seizures: List[Tuple[float, float]],
    ) -> Tuple[List[Alarm], List[SeizureEvent], float]:
        """
        Evaluate a single recording.
        
        Args:
            risk_scores: Risk scores per window [N]
            window_times: End time of each window [N] (seconds)
            seizures: List of (onset, offset) tuples
            
        Returns:
            Tuple of (alarms, seizure_events, duration_hours)
        """
        # Find alarm triggers
        alarms = self._find_alarms(risk_scores, window_times)
        
        # Evaluate each seizure
        seizure_events = []
        for onset, offset in seizures:
            event = self._evaluate_seizure(onset, offset, alarms)
            seizure_events.append(event)
        
        # Mark true/false alarms
        for alarm in alarms:
            alarm.is_true_alarm = self._is_true_alarm(alarm.time_sec, seizures)
        
        # Recording duration
        duration_sec = window_times[-1] if len(window_times) > 0 else 0
        duration_hours = duration_sec / 3600
        
        return alarms, seizure_events, duration_hours
    
    def _find_alarms(
        self,
        risk_scores: np.ndarray,
        window_times: np.ndarray,
    ) -> List[Alarm]:
        """Find alarm times applying refractory period."""
        alarms = []
        last_alarm_time = -np.inf
        
        for i, (score, time) in enumerate(zip(risk_scores, window_times)):
            if score >= self.threshold:
                # Check refractory period
                if time - last_alarm_time >= self.refractory_sec:
                    alarms.append(Alarm(
                        time_sec=time,
                        risk_score=score,
                        is_true_alarm=False,  # Will be set later
                    ))
                    last_alarm_time = time
        
        return alarms
    
    def _evaluate_seizure(
        self,
        onset: float,
        offset: float,
        alarms: List[Alarm],
    ) -> SeizureEvent:
        """Check if seizure was predicted by any alarm."""
        # Valid alarm window: [onset - preictal_sec, onset - gap_sec]
        alarm_window_start = onset - self.preictal_sec
        alarm_window_end = onset - self.gap_sec
        
        # Find alarms in window
        valid_alarms = [
            a for a in alarms
            if alarm_window_start <= a.time_sec < alarm_window_end
        ]
        
        if valid_alarms:
            first_alarm = min(valid_alarms, key=lambda a: a.time_sec)
            warning_time = onset - first_alarm.time_sec
            return SeizureEvent(
                onset_sec=onset,
                offset_sec=offset,
                was_predicted=True,
                first_alarm_time=first_alarm.time_sec,
                warning_time=warning_time,
            )
        else:
            return SeizureEvent(
                onset_sec=onset,
                offset_sec=offset,
                was_predicted=False,
                first_alarm_time=None,
                warning_time=None,
            )
    
    def _is_true_alarm(
        self,
        alarm_time: float,
        seizures: List[Tuple[float, float]],
    ) -> bool:
        """Check if alarm correctly precedes a seizure."""
        for onset, offset in seizures:
            alarm_window_start = onset - self.preictal_sec
            alarm_window_end = onset - self.gap_sec
            
            if alarm_window_start <= alarm_time < alarm_window_end:
                return True
        
        return False
    
    @staticmethod
    def aggregate_results(
        all_alarms: List[List[Alarm]],
        all_seizures: List[List[SeizureEvent]],
        all_durations: List[float],
    ) -> AlarmMetrics:
        """
        Aggregate results across multiple recordings.
        
        Args:
            all_alarms: List of alarm lists per recording
            all_seizures: List of seizure event lists per recording
            all_durations: Duration of each recording (hours)
            
        Returns:
            Aggregated AlarmMetrics
        """
        total_duration = sum(all_durations)
        
        n_seizures = sum(len(s) for s in all_seizures)
        n_predicted = sum(
            sum(1 for e in events if e.was_predicted)
            for events in all_seizures
        )
        
        n_false_alarms = sum(
            sum(1 for a in alarms if not a.is_true_alarm)
            for alarms in all_alarms
        )
        
        # Collect warning times
        warning_times = []
        for events in all_seizures:
            for e in events:
                if e.warning_time is not None:
                    warning_times.append(e.warning_time)
        
        mean_warning = np.mean(warning_times) if warning_times else 0.0
        
        # FAH
        fah = n_false_alarms / max(total_duration, 1e-6)
        
        # Sensitivity
        sensitivity = n_predicted / max(n_seizures, 1)
        
        return AlarmMetrics(
            false_alarms_per_hour=fah,
            sensitivity=sensitivity,
            mean_warning_time=mean_warning,
            n_seizures=n_seizures,
            n_predicted=n_predicted,
            n_false_alarms=n_false_alarms,
            total_duration_hours=total_duration,
        )


def compute_alarm_metrics(
    risk_scores: List[np.ndarray],
    window_times: List[np.ndarray],
    seizures_per_recording: List[List[Tuple[float, float]]],
    threshold: float = 0.5,
    refractory_min: float = 20.0,
    preictal_min: float = 10.0,
    gap_sec: float = 30.0,
) -> AlarmMetrics:
    """
    Compute alarm metrics across multiple recordings.
    
    Args:
        risk_scores: List of risk score arrays per recording
        window_times: List of window end times per recording
        seizures_per_recording: List of seizure lists per recording
        threshold: Alarm threshold
        refractory_min: Refractory period (minutes)
        preictal_min: Preictal horizon (minutes)
        gap_sec: Gap before onset (seconds)
        
    Returns:
        AlarmMetrics
    """
    evaluator = AlarmEvaluator(
        threshold=threshold,
        refractory_min=refractory_min,
        preictal_min=preictal_min,
        gap_sec=gap_sec,
    )
    
    all_alarms = []
    all_seizures = []
    all_durations = []
    
    for scores, times, seizures in zip(
        risk_scores, window_times, seizures_per_recording
    ):
        alarms, events, duration = evaluator.evaluate_recording(
            scores, times, seizures
        )
        all_alarms.append(alarms)
        all_seizures.append(events)
        all_durations.append(duration)
    
    return AlarmEvaluator.aggregate_results(
        all_alarms, all_seizures, all_durations
    )


def evaluate_at_thresholds(
    risk_scores: List[np.ndarray],
    window_times: List[np.ndarray],
    seizures_per_recording: List[List[Tuple[float, float]]],
    thresholds: List[float],
    cfg: Optional[DictConfig] = None,
) -> Dict[float, AlarmMetrics]:
    """
    Evaluate at multiple thresholds.
    
    Returns:
        Dict mapping threshold to AlarmMetrics
    """
    refractory_min = cfg.evaluation.refractory_min if cfg else 20.0
    preictal_min = cfg.windowing.preictal_min if cfg else 10.0
    gap_sec = cfg.windowing.gap_sec if cfg else 30.0
    
    results = {}
    for thresh in thresholds:
        metrics = compute_alarm_metrics(
            risk_scores=risk_scores,
            window_times=window_times,
            seizures_per_recording=seizures_per_recording,
            threshold=thresh,
            refractory_min=refractory_min,
            preictal_min=preictal_min,
            gap_sec=gap_sec,
        )
        results[thresh] = metrics
    
    return results
