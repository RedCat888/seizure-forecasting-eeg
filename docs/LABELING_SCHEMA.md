# Labeling Schema for Seizure Forecasting

## Overview

This document describes the labeling strategy for seizure **forecasting** (prediction before onset), as opposed to seizure **detection** (identification during ictal phase).

The goal is to train a classifier that can distinguish **preictal** (pre-seizure) windows from **interictal** (normal baseline) windows, enabling early warning before seizure onset.

## Key Concepts

### Seizure Phases

```
     INTERICTAL          PREICTAL       GAP   ICTAL      POSTICTAL      INTERICTAL
  (normal baseline)   (pre-seizure)   (exc) (seizure)  (recovery)    (return to baseline)
  
─────────────────────┬───────────────┬─────┬──────────┬─────────────┬────────────────────
                     │←─ 10 min ────→│30s │          │←─ 10 min ──→│
                     │               │    │          │             │
                     └───────────────┘    │          │             │
                         ↑                ↑          ↑
                    PREICTAL START    GAP START   SEIZURE ONSET
```

### Window Classification

Each EEG window is assigned one of three labels:

1. **PREICTAL (1)**: Windows in the danger zone before seizure
2. **INTERICTAL (0)**: Normal baseline windows, far from any seizure
3. **EXCLUDED**: Windows that are ambiguous or during/after seizure

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_SFREQ` | 256 Hz | Sampling frequency after resampling |
| `W_SEC` | 10.0 s | Window length in seconds |
| `STEP_SEC` | 5.0 s | Window stride (50% overlap) |
| `PREICTAL_MIN` | 10 min | Preictal horizon before seizure |
| `GAP_SEC` | 30 s | Exclusion gap immediately before onset |
| `POSTICTAL_MIN` | 10 min | Post-seizure exclusion period |
| `INTERICTAL_BUFFER_MIN` | 30 min | Minimum distance from seizure for interictal |
| `TAU_SEC` | 120 s | Time constant for soft risk decay |

## Labeling Rules

### Rule 1: Preictal Windows

A window is labeled **PREICTAL (y_cls = 1)** if:

```
window_end ∈ [onset - PREICTAL_MIN*60, onset - GAP_SEC]
```

In other words:
- Window ends at least `GAP_SEC` (30s) before seizure onset
- Window ends at most `PREICTAL_MIN*60` (600s) before seizure onset

**Example:**
- Seizure onset at t=1000s
- Preictal zone: [1000-600, 1000-30] = [400s, 970s]
- A window ending at t=500s → PREICTAL
- A window ending at t=990s → EXCLUDED (in gap)

### Rule 2: Excluded Windows

A window is **EXCLUDED** if any of these conditions apply:

1. **In gap zone**: `window_end ∈ [onset - GAP_SEC, onset]`
2. **During seizure (ictal)**: `window overlaps [onset, offset]`
3. **In postictal recovery**: `window_start < offset + POSTICTAL_MIN*60`
4. **Too close to seizure**: Not preictal but within buffer zone

### Rule 3: Interictal Windows

A window is labeled **INTERICTAL (y_cls = 0)** if:

- It is NOT preictal for any seizure
- It is NOT excluded for any reason
- Its center is at least `INTERICTAL_BUFFER_MIN` minutes away from any seizure onset or offset

```python
for each seizure (onset, offset):
    if abs(window_center - onset) < INTERICTAL_BUFFER_MIN * 60:
        → NOT interictal
    if abs(window_center - offset) < INTERICTAL_BUFFER_MIN * 60:
        → NOT interictal
```

## Output Labels

For each valid (non-excluded) window, we compute:

### 1. Binary Classification Label (y_cls)

```python
y_cls = 1 if preictal else 0
```

### 2. Time-to-Event (y_tte)

For preictal windows, the time until seizure onset:

```python
if preictal:
    y_tte = seizure_onset - window_end  # seconds
else:
    y_tte = -1  # sentinel for interictal
```

### 3. Soft Risk Score (y_soft)

A continuous risk value that increases approaching seizure:

```python
if preictal:
    t = seizure_onset - window_end  # time until onset
    y_soft = exp(-t / TAU_SEC)  # exponential rise
else:
    y_soft = 0.0
```

**Soft risk profile:**

| Time before onset | y_soft value |
|-------------------|--------------|
| 10 min (600s) | 0.0067 |
| 5 min (300s) | 0.082 |
| 2 min (120s) | 0.368 |
| 1 min (60s) | 0.607 |
| 30s (gap start) | 0.779 |

## Handling Multiple Seizures

When a recording has multiple seizures:

1. A window may be preictal for the nearest upcoming seizure
2. Postictal exclusion from seizure N may overlap preictal zone for seizure N+1
3. In such cases, the window is **EXCLUDED** (ambiguous)

```python
def check_window(window_start, window_end, seizures):
    # Check each seizure
    for onset, offset in seizures:
        # Exclusions take priority
        if is_excluded(window_start, window_end, onset, offset):
            return 'excluded', None
    
    # Check preictal status
    for onset, offset in seizures:
        if is_preictal(window_end, onset):
            tte = onset - window_end
            return 'preictal', tte
    
    # Check interictal
    if is_interictal(window_start, window_end, seizures):
        return 'interictal', None
    
    return 'excluded', None  # Falls in buffer zone
```

## Class Imbalance

The dataset has significant class imbalance:

- **Interictal**: ~95-98% of windows
- **Preictal**: ~2-5% of windows

Mitigation strategies:
1. **Weighted loss**: `pos_weight = n_neg / n_pos`
2. **Sample weighting**: Higher weight for windows closer to onset
3. **Balanced sampling**: Oversample preictal during training

## Sample Weighting

For preictal windows, we apply additional weighting to emphasize windows closer to seizure onset:

```python
if preictal:
    t = onset - window_end  # seconds until onset
    max_t = PREICTAL_MIN * 60  # 600 seconds
    
    # Higher weight for windows closer to onset
    w = 1 + (1 - t / max_t)  # Range: [1, 2]
else:
    w = 1.0
```

## Visual Timeline Example

```
Recording: chb01_03.edf (1 hour, 1 seizure at 30 min)

Time:  0    5    10   15   20   25   30   35   40   45   50   55   60 min
       │    │    │    │    │    │    │    │    │    │    │    │    │
       ├────┴────┴────┴────┤    ├────┤    ├────┴────┴────┴────┴────┤
       │                   │    │    │    │                        │
       │   INTERICTAL      │ PRE│GAP │POST│      INTERICTAL        │
       │                   │ICTAL│EXC│ICTAL│                       │
       └───────────────────┴────┴────┴────┴────────────────────────┘
                           │    │    │    │
                           20   30   30   40
                          min  min  min   min
                               ↑
                          SEIZURE ONSET

Legend:
- INTERICTAL: Windows labeled 0, used for training
- PREICTAL: Windows labeled 1, 10-minute zone before seizure
- GAP: 30-second exclusion zone, too close to onset
- POSTICTAL: 10-minute recovery period, excluded
```

## Implementation Notes

### File-Level Processing

1. Parse summary file to get seizure times for each EDF
2. Load EDF and preprocess
3. Slide windows across the recording
4. For each window, compute labels using rules above
5. Store valid windows with metadata

### Edge Cases

1. **Seizure at recording start**: No preictal windows available
2. **Seizure at recording end**: Postictal extends beyond file
3. **Back-to-back seizures**: Windows between may be excluded
4. **No seizures in file**: All windows are interictal candidates

### Validation

Always verify:
- No window overlaps ictal period
- Preictal and interictal windows don't overlap
- Time-to-event is always positive for preictal windows
- Soft risk increases monotonically approaching seizure
