#!/usr/bin/env python3
"""
Label Sanity Check Visualization.

Creates plots showing EEG data with labeled regions:
- Preictal windows (orange)
- GAP region (gray)  
- Ictal (seizure) region (red)
- Postictal exclusion (gray hatched)
- Interictal (green)

Seizure onset marked with vertical red line.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.chbmit.parse_summary import get_seizure_times
from src.signal.preprocess import preprocess_edf, load_edf_raw, get_eeg_channels


def plot_label_sanity(
    edf_path: Path,
    seizures: list,
    output_path: Path,
    cfg,
    time_window_sec: float = 900,  # 15 minutes around seizure
    n_channels: int = 6,
):
    """
    Create a label sanity visualization for a seizure file.
    
    Args:
        edf_path: Path to EDF file
        seizures: List of (onset, offset) tuples
        output_path: Where to save the plot
        cfg: Configuration object
        time_window_sec: Total time window to show
        n_channels: Number of channels to display
    """
    if not seizures:
        print(f"  No seizures in {edf_path.name}, skipping")
        return False
    
    # Get labeling parameters from config
    preictal_min = cfg.windowing.preictal_min
    gap_sec = cfg.windowing.gap_sec
    postictal_min = cfg.windowing.postictal_min
    interictal_buffer_min = cfg.windowing.interictal_buffer_min
    
    preictal_sec = preictal_min * 60
    postictal_sec = postictal_min * 60
    interictal_buffer_sec = interictal_buffer_min * 60
    
    # Load and preprocess EEG
    print(f"  Loading {edf_path.name}...")
    
    try:
        data, channels, sfreq = preprocess_edf(edf_path, cfg=cfg, verbose=False)
    except Exception as e:
        print(f"  Error preprocessing: {e}")
        return False
    
    recording_duration = data.shape[1] / sfreq
    
    # Use first seizure as center
    seizure_onset, seizure_offset = seizures[0]
    seizure_duration = seizure_offset - seizure_onset
    
    # Center window around seizure onset
    center = seizure_onset
    start_time = max(0, center - time_window_sec / 2)
    end_time = min(recording_duration, start_time + time_window_sec)
    
    # Adjust start if we hit the end
    if end_time - start_time < time_window_sec:
        start_time = max(0, end_time - time_window_sec)
    
    # Extract data segment
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)
    segment = data[:n_channels, start_sample:end_sample]
    times = np.arange(segment.shape[1]) / sfreq + start_time
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot EEG traces
    n_ch = min(n_channels, segment.shape[0])
    ch_spacing = np.std(segment) * 5  # Spacing between channels
    
    for i in range(n_ch):
        offset = (n_ch - i - 1) * ch_spacing
        ax.plot(times, segment[i] * 1e6 + offset * 1e6, 'k', linewidth=0.3, alpha=0.8)
    
    # Add channel labels
    y_positions = [(n_ch - i - 1) * ch_spacing * 1e6 for i in range(n_ch)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(channels[:n_ch])
    
    # Define label regions relative to first seizure
    regions = []
    
    # 1. Preictal region (before gap)
    preictal_start = seizure_onset - preictal_sec
    preictal_end = seizure_onset - gap_sec
    if preictal_end > start_time and preictal_start < end_time:
        regions.append({
            'start': max(preictal_start, start_time),
            'end': min(preictal_end, end_time),
            'color': '#f97316',  # Orange
            'alpha': 0.3,
            'label': 'Preictal'
        })
    
    # 2. GAP region
    gap_start = seizure_onset - gap_sec
    gap_end = seizure_onset
    if gap_end > start_time and gap_start < end_time:
        regions.append({
            'start': max(gap_start, start_time),
            'end': min(gap_end, end_time),
            'color': '#94a3b8',  # Gray
            'alpha': 0.4,
            'label': 'Gap (excluded)'
        })
    
    # 3. Ictal region (during seizure)
    if seizure_offset > start_time and seizure_onset < end_time:
        regions.append({
            'start': max(seizure_onset, start_time),
            'end': min(seizure_offset, end_time),
            'color': '#ef4444',  # Red
            'alpha': 0.4,
            'label': 'Ictal (seizure)'
        })
    
    # 4. Postictal region
    postictal_start = seizure_offset
    postictal_end = seizure_offset + postictal_sec
    if postictal_end > start_time and postictal_start < end_time:
        regions.append({
            'start': max(postictal_start, start_time),
            'end': min(postictal_end, end_time),
            'color': '#94a3b8',  # Gray
            'alpha': 0.3,
            'label': 'Postictal (excluded)',
            'hatch': '//'
        })
    
    # 5. Interictal buffer before preictal
    buffer_start = preictal_start - interictal_buffer_sec
    buffer_end = preictal_start
    if buffer_end > start_time and buffer_start < end_time:
        regions.append({
            'start': max(buffer_start, start_time),
            'end': min(buffer_end, end_time),
            'color': '#94a3b8',
            'alpha': 0.15,
            'label': 'Buffer (excluded)',
            'hatch': '..'
        })
    
    # 6. Interictal region (mark remaining as interictal - far from seizure)
    # This is implicit - areas not covered by other regions
    
    # Get y-axis limits
    ylim = ax.get_ylim()
    
    # Draw shaded regions
    for r in regions:
        rect = Rectangle(
            (r['start'], ylim[0]),
            r['end'] - r['start'],
            ylim[1] - ylim[0],
            facecolor=r['color'],
            alpha=r['alpha'],
            edgecolor='none',
            hatch=r.get('hatch', None),
            zorder=0
        )
        ax.add_patch(rect)
    
    # Draw seizure onset vertical line
    if start_time <= seizure_onset <= end_time:
        ax.axvline(seizure_onset, color='#dc2626', linewidth=2, linestyle='--', label='Seizure Onset')
    
    # Draw seizure offset line
    if start_time <= seizure_offset <= end_time:
        ax.axvline(seizure_offset, color='#dc2626', linewidth=1.5, linestyle=':', alpha=0.7)
    
    # Add time markers
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    ax.set_xlim([start_time, end_time])
    
    # Title with seizure info
    ax.set_title(
        f'Label Sanity Check: {edf_path.name}\n'
        f'Seizure onset: {seizure_onset:.1f}s, duration: {seizure_duration:.1f}s | '
        f'Preictal: {preictal_min}min, Gap: {gap_sec}s, Postictal: {postictal_min}min',
        fontsize=11
    )
    
    # Create legend
    legend_elements = [
        Patch(facecolor='#22c55e', alpha=0.3, label='Interictal'),
        Patch(facecolor='#f97316', alpha=0.3, label=f'Preictal ({preictal_min}min before)'),
        Patch(facecolor='#94a3b8', alpha=0.4, label=f'Gap ({gap_sec}s excluded)'),
        Patch(facecolor='#ef4444', alpha=0.4, label='Ictal (seizure)'),
        Patch(facecolor='#94a3b8', alpha=0.3, hatch='//', label=f'Postictal ({postictal_min}min excluded)'),
        Line2D([0], [0], color='#dc2626', linewidth=2, linestyle='--', label='Seizure Onset'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add text annotations for region boundaries
    text_y = ylim[1] * 0.95
    
    if start_time <= preictal_start <= end_time:
        ax.axvline(preictal_start, color='#f97316', linewidth=1, linestyle='-', alpha=0.5)
        ax.text(preictal_start, text_y, f'Preictal\nstart', fontsize=8, ha='center', color='#f97316')
    
    if start_time <= preictal_end <= end_time:
        ax.axvline(preictal_end, color='#f97316', linewidth=1, linestyle='-', alpha=0.5)
    
    if start_time <= postictal_end <= end_time:
        ax.axvline(postictal_end, color='#94a3b8', linewidth=1, linestyle='-', alpha=0.5)
        ax.text(postictal_end, text_y, f'Postictal\nend', fontsize=8, ha='center', color='#666')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    return True


def find_seizure_files(data_root: Path, subject: str, max_files: int = 1):
    """Find EDF files with seizures for a subject."""
    seizures_by_file = get_seizure_times(data_root, subject)
    
    seizure_files = []
    for fname, seizures in seizures_by_file.items():
        if len(seizures) > 0:
            edf_path = data_root / subject / fname
            if edf_path.exists():
                seizure_files.append((edf_path, seizures))
                if len(seizure_files) >= max_files:
                    break
    
    return seizure_files


def main():
    parser = argparse.ArgumentParser(description="Generate label sanity visualizations")
    parser.add_argument("--config", type=str, default="configs/small_run.yaml")
    parser.add_argument("--data_root", type=str, default="data/chbmit_raw")
    parser.add_argument("--output_dir", type=str, default="reports/figures")
    parser.add_argument("--subjects", type=str, nargs="+", default=["chb01", "chb02", "chb03"])
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LABEL SANITY CHECK VISUALIZATION")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {args.subjects}")
    print()
    print("Labeling parameters:")
    print(f"  Preictal window: {cfg.windowing.preictal_min} minutes before onset")
    print(f"  Gap (excluded): {cfg.windowing.gap_sec} seconds before onset")
    print(f"  Postictal (excluded): {cfg.windowing.postictal_min} minutes after offset")
    print(f"  Interictal buffer: {cfg.windowing.interictal_buffer_min} minutes from seizure")
    print()
    
    plots_created = 0
    
    for subject in args.subjects:
        print(f"\n[{subject}]")
        
        try:
            seizure_files = find_seizure_files(data_root, subject, max_files=1)
        except Exception as e:
            print(f"  Error finding seizure files: {e}")
            continue
        
        if not seizure_files:
            print(f"  No seizure files found")
            continue
        
        for edf_path, seizures in seizure_files:
            output_path = output_dir / f"label_sanity_{subject}_{edf_path.stem}.png"
            
            success = plot_label_sanity(
                edf_path=edf_path,
                seizures=seizures,
                output_path=output_path,
                cfg=cfg,
                time_window_sec=900,  # 15 minutes
                n_channels=6,
            )
            
            if success:
                plots_created += 1
    
    print("\n" + "=" * 60)
    print(f"Created {plots_created} label sanity plots")
    print("=" * 60)
    
    if plots_created == 0:
        print("WARNING: No plots created! Check data paths.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
