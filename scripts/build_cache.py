#!/usr/bin/env python3
"""
Build preprocessed window cache from CHB-MIT dataset.

This script:
1. Loads EDF files for specified subjects
2. Preprocesses (filter, resample, re-reference)
3. ENFORCES CANONICAL CHANNEL ORDER (identical across all samples)
4. Extracts windows with labels (preictal/interictal)
5. Computes handcrafted features
6. Saves to HDF5 or Cache V2 (memmap) format
7. Verifies channel consistency

Usage:
    # Standard HDF5 format (one file per subject)
    python scripts/build_cache.py --data_root data/chbmit_raw --out_root data/chbmit_cache --subjects chb01 chb02 chb03
    
    # Cache V2 format (memmap, RAM-stable)
    python scripts/build_cache.py --data_root data/chbmit_raw --out_root data/chbmit_cache_v2 --subjects all --cache_format v2 --dtype float16
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import h5py
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.chbmit import get_subject_list, get_edf_files, get_seizure_times
from src.signal.preprocess import preprocess_edf, check_window_artifact, CANONICAL_CHANNELS
from src.features.handcrafted import extract_features, get_feature_names
from src.data.windowing import generate_windows, WindowInfo
from src.data.cache_v2 import CacheV2Builder


console = Console()


def process_subject(
    subject: str,
    data_root: Path,
    out_root: Path,
    cfg,
    channel_log_file: Optional[Path] = None,
) -> dict:
    """
    Process all EDF files for a subject with canonical channel ordering.
    
    Returns:
        Dict with counts and channel info
    """
    console.print(f"\n[bold cyan]Processing {subject}[/bold cyan]")
    
    subject_dir = data_root / subject
    out_file = out_root / f"{subject}.h5"
    
    # Get seizure times for this subject
    try:
        seizures_by_file = get_seizure_times(data_root, subject)
    except FileNotFoundError:
        console.print(f"  [yellow]Warning: No summary file found for {subject}[/yellow]")
        seizures_by_file = {}
    
    # Get EDF files
    edf_files = get_edf_files(data_root, subject)
    
    if not edf_files:
        console.print(f"  [yellow]No EDF files found for {subject}[/yellow]")
        return {"n_windows": 0, "n_preictal": 0, "n_interictal": 0, "channel_info": None}
    
    console.print(f"  Found {len(edf_files)} EDF files")
    console.print(f"  [dim]Using CANONICAL_CHANNELS: {len(CANONICAL_CHANNELS)} channels[/dim]")
    
    # Storage for all windows
    all_data = []
    all_features = []
    all_y_cls = []
    all_y_tte = []
    all_y_soft = []
    all_metadata = []
    channel_stats = {"total_files": 0, "files_with_missing": 0, "missing_channels": set()}
    
    # Windowing parameters
    w_cfg = cfg.windowing
    s_cfg = cfg.signal
    
    window_samples = int(w_cfg.window_sec * s_cfg.target_sfreq)
    
    # Expected number of channels
    expected_n_channels = len(CANONICAL_CHANNELS)
    
    for edf_name in tqdm(edf_files, desc=f"  {subject}", leave=False):
        edf_path = subject_dir / edf_name
        
        try:
            # Preprocess EDF with canonical channel enforcement
            data, channels, sfreq, channel_info = preprocess_edf(
                edf_path, 
                cfg=cfg, 
                enforce_canonical=True,
                channel_log_file=channel_log_file,
            )
            
            # Verify channel consistency
            assert data.shape[0] == expected_n_channels, \
                f"Channel mismatch in {edf_name}: expected {expected_n_channels}, got {data.shape[0]}"
            assert channels == CANONICAL_CHANNELS, \
                f"Channel order mismatch in {edf_name}"
            
            # Track channel statistics
            channel_stats["total_files"] += 1
            if channel_info and channel_info.get("n_missing", 0) > 0:
                channel_stats["files_with_missing"] += 1
                channel_stats["missing_channels"].update(channel_info.get("missing_channels", []))
                
        except Exception as e:
            console.print(f"  [red]Error processing {edf_name}: {e}[/red]")
            continue
        
        # Get seizures for this file
        seizures = seizures_by_file.get(edf_name, [])
        
        # Recording duration
        recording_duration = data.shape[1] / sfreq
        
        # Generate windows
        windows = list(generate_windows(
            recording_duration_sec=recording_duration,
            sfreq=sfreq,
            seizures=seizures,
            window_sec=w_cfg.window_sec,
            step_sec=w_cfg.step_sec,
            preictal_min=w_cfg.preictal_min,
            gap_sec=w_cfg.gap_sec,
            postictal_min=w_cfg.postictal_min,
            interictal_buffer_min=w_cfg.interictal_buffer_min,
            tau_sec=w_cfg.tau_sec,
        ))
        
        for win_info in windows:
            # Skip excluded windows
            if win_info.label == "excluded":
                continue
            
            # Extract window data
            window_data = data[:, win_info.start_sample:win_info.end_sample]
            
            # Check for correct size
            if window_data.shape[1] != window_samples:
                continue
            
            # Artifact rejection
            if not check_window_artifact(window_data, s_cfg.amp_uv_thresh):
                continue
            
            # Extract features
            features = extract_features(window_data, sfreq, cfg)
            
            # Store
            all_data.append(window_data.astype(np.float32))
            all_features.append(features)
            all_y_cls.append(win_info.y_cls)
            all_y_tte.append(win_info.y_tte)
            all_y_soft.append(win_info.y_soft)
            all_metadata.append({
                "file": edf_name,
                "start_sec": win_info.start_sec,
                "end_sec": win_info.end_sec,
            })
    
    if not all_data:
        console.print(f"  [yellow]No valid windows for {subject}[/yellow]")
        return {"n_windows": 0, "n_preictal": 0, "n_interictal": 0, "channel_info": channel_stats}
    
    # Stack arrays
    data_array = np.stack(all_data, axis=0)
    features_array = np.stack(all_features, axis=0)
    y_cls_array = np.array(all_y_cls, dtype=np.int32)
    y_tte_array = np.array(all_y_tte, dtype=np.float32)
    y_soft_array = np.array(all_y_soft, dtype=np.float32)
    
    # FINAL VERIFICATION: All windows have identical channel count
    assert data_array.shape[1] == expected_n_channels, \
        f"Final channel verification failed: expected {expected_n_channels}, got {data_array.shape[1]}"
    
    # Count labels
    n_preictal = int(np.sum(y_cls_array == 1))
    n_interictal = int(np.sum(y_cls_array == 0))
    
    # Save to HDF5
    out_root.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(out_file, "w") as f:
        f.create_dataset("data", data=data_array, compression="gzip")
        f.create_dataset("features", data=features_array, compression="gzip")
        f.create_dataset("y_cls", data=y_cls_array)
        f.create_dataset("y_tte", data=y_tte_array)
        f.create_dataset("y_soft", data=y_soft_array)
        
        # Store CANONICAL channel names (always the same order)
        f.attrs["channels"] = ",".join(CANONICAL_CHANNELS)
        f.attrs["sfreq"] = sfreq
        f.attrs["subject"] = subject
        f.attrs["n_channels"] = expected_n_channels
        
        # Store feature names
        feature_names = get_feature_names(cfg)
        f.attrs["feature_names"] = ",".join(feature_names)
    
    # Log channel statistics
    if channel_stats["files_with_missing"] > 0:
        console.print(
            f"  [yellow]Channel warning: {channel_stats['files_with_missing']}/{channel_stats['total_files']} "
            f"files had missing channels (filled with zeros)[/yellow]"
        )
    
    console.print(
        f"  Saved {len(all_data)} windows "
        f"([green]preictal: {n_preictal}[/green], "
        f"[blue]interictal: {n_interictal}[/blue]) "
        f"[dim]shape: ({data_array.shape})[/dim]"
    )
    
    return {
        "n_windows": len(all_data),
        "n_preictal": n_preictal,
        "n_interictal": n_interictal,
        "channel_info": channel_stats,
    }


def verify_cache_channel_consistency(cache_dir: Path):
    """
    Verify all cached files have identical channel order.
    
    Args:
        cache_dir: Path to cache directory
        
    Returns:
        True if consistent, raises AssertionError otherwise
    """
    h5_files = list(cache_dir.glob("*.h5"))
    if not h5_files:
        console.print("[yellow]No cache files to verify[/yellow]")
        return True
    
    console.print(f"\n[bold]Verifying channel consistency across {len(h5_files)} cache files...[/bold]")
    
    reference_channels = None
    reference_n_channels = None
    
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            channels = f.attrs.get("channels", "").split(",")
            n_channels = f.attrs.get("n_channels", f["data"].shape[1])
            data_shape = f["data"].shape
            
            if reference_channels is None:
                reference_channels = channels
                reference_n_channels = n_channels
                console.print(f"  Reference: {n_channels} channels from {h5_path.name}")
            else:
                # Verify
                assert channels == reference_channels, \
                    f"Channel order mismatch in {h5_path.name}: {channels[:3]}... vs {reference_channels[:3]}..."
                assert data_shape[1] == reference_n_channels, \
                    f"Channel count mismatch in {h5_path.name}: {data_shape[1]} vs {reference_n_channels}"
    
    console.print(f"  [green]OK: All {len(h5_files)} files have identical channel order: {reference_n_channels} channels[/green]")
    console.print(f"  [dim]Channels: {', '.join(reference_channels[:6])}...[/dim]")
    
    return True


def estimate_total_windows(
    subjects: List[str],
    data_root: Path,
    cfg,
) -> int:
    """Estimate total windows across all subjects for V2 allocation."""
    total = 0
    w_cfg = cfg.windowing
    s_cfg = cfg.signal
    
    for subject in subjects:
        edf_files = get_edf_files(data_root, subject)
        for edf_name in edf_files:
            edf_path = data_root / subject / edf_name
            if edf_path.exists():
                # Rough estimate: ~1 hour per file, 30s windows, 15s stride
                # ~240 windows per file
                total += 240
    
    # Add 20% buffer for safety
    return int(total * 1.2)


def build_cache_v2(
    subjects: List[str],
    data_root: Path,
    out_root: Path,
    cfg,
    dtype: str = "float16",
) -> dict:
    """
    Build Cache V2 (memmap format) for all subjects.
    
    This creates a single set of memmap files containing all windows
    from all subjects, enabling RAM-stable training.
    
    Args:
        subjects: List of subject IDs
        data_root: Path to raw data
        out_root: Output directory for cache
        cfg: Config
        dtype: Data type for storage ("float16" or "float32")
        
    Returns:
        Dict with statistics
    """
    console.print("\n[bold cyan]Building Cache V2 (memmap format)[/bold cyan]")
    console.print(f"  Dtype: {dtype}")
    console.print(f"  Subjects: {len(subjects)}")
    
    # Windowing parameters
    w_cfg = cfg.windowing
    s_cfg = cfg.signal
    window_samples = int(w_cfg.window_sec * s_cfg.target_sfreq)
    n_channels = len(CANONICAL_CHANNELS)
    
    # Get feature dimension
    feature_names = get_feature_names(cfg)
    n_features = len(feature_names)
    
    # Estimate total windows
    estimated_windows = estimate_total_windows(subjects, data_root, cfg)
    console.print(f"  Estimated windows: {estimated_windows}")
    
    # Initialize builder
    builder = CacheV2Builder(
        output_dir=out_root,
        n_channels=n_channels,
        window_len=window_samples,
        n_features=n_features,
        dtype=dtype,
        include_spectrograms=False,  # Skip spectrograms for now
    )
    
    builder.begin(estimated_samples=estimated_windows)
    
    # Process all subjects
    total_stats = {"n_windows": 0, "n_preictal": 0, "n_interictal": 0, "n_excluded": 0}
    seizure_id_counter = 0
    
    with Progress() as progress:
        subject_task = progress.add_task("[cyan]Processing subjects...", total=len(subjects))
        
        for subject in subjects:
            subject_dir = data_root / subject
            
            # Get seizure times
            try:
                seizures_by_file = get_seizure_times(data_root, subject)
            except FileNotFoundError:
                console.print(f"  [yellow]Warning: No summary file for {subject}[/yellow]")
                seizures_by_file = {}
            
            edf_files = get_edf_files(data_root, subject)
            
            for edf_name in edf_files:
                edf_path = subject_dir / edf_name
                
                try:
                    # Preprocess
                    data, channels, sfreq, channel_info = preprocess_edf(
                        edf_path, cfg=cfg, enforce_canonical=True
                    )
                    
                    if data.shape[0] != n_channels:
                        continue
                        
                except Exception as e:
                    console.print(f"  [red]Error: {edf_name}: {e}[/red]")
                    continue
                
                # Get seizures for this file
                seizures = seizures_by_file.get(edf_name, [])
                recording_duration = data.shape[1] / sfreq
                
                # Generate windows
                windows = list(generate_windows(
                    recording_duration_sec=recording_duration,
                    sfreq=sfreq,
                    seizures=seizures,
                    window_sec=w_cfg.window_sec,
                    step_sec=w_cfg.step_sec,
                    preictal_min=w_cfg.preictal_min,
                    gap_sec=w_cfg.gap_sec,
                    postictal_min=w_cfg.postictal_min,
                    interictal_buffer_min=w_cfg.interictal_buffer_min,
                    tau_sec=w_cfg.tau_sec,
                ))
                
                for win_info in windows:
                    if win_info.label == "excluded":
                        total_stats["n_excluded"] += 1
                        continue
                    
                    window_data = data[:, win_info.start_sample:win_info.end_sample]
                    
                    if window_data.shape[1] != window_samples:
                        continue
                    
                    if not check_window_artifact(window_data, s_cfg.amp_uv_thresh):
                        continue
                    
                    # Extract features
                    features = extract_features(window_data, sfreq, cfg)
                    
                    # Add to cache
                    builder.add_sample(
                        data=window_data,
                        features=features,
                        y_cls=win_info.y_cls,
                        y_tte=win_info.y_tte,
                        y_soft=win_info.y_soft,
                        seizure_id=seizure_id_counter if win_info.y_cls == 1 else -1,
                        subject=subject,
                        edf_path=edf_name,
                        start_sec=win_info.start_sec,
                        end_sec=win_info.end_sec,
                    )
                    
                    total_stats["n_windows"] += 1
                    if win_info.y_cls == 1:
                        total_stats["n_preictal"] += 1
                    else:
                        total_stats["n_interictal"] += 1
                
                # Increment seizure ID after each seizure file
                if seizures:
                    seizure_id_counter += 1
            
            progress.update(subject_task, advance=1)
    
    # Finalize
    metadata = builder.finalize()
    
    # Save window counts report
    reports_dir = Path("reports/tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    window_counts_path = reports_dir / "window_counts_full.csv"
    with open(window_counts_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"n_windows,{total_stats['n_windows']}\n")
        f.write(f"n_preictal,{total_stats['n_preictal']}\n")
        f.write(f"n_interictal,{total_stats['n_interictal']}\n")
        f.write(f"n_excluded,{total_stats['n_excluded']}\n")
        f.write(f"preictal_pct,{100.0 * total_stats['n_preictal'] / max(total_stats['n_windows'], 1):.2f}\n")
        f.write(f"n_subjects,{len(subjects)}\n")
    
    console.print(f"\n[bold green]Cache V2 built successfully![/bold green]")
    console.print(f"  Total windows: {total_stats['n_windows']}")
    console.print(f"  Preictal: {total_stats['n_preictal']} ({100.0 * total_stats['n_preictal'] / max(total_stats['n_windows'], 1):.2f}%)")
    console.print(f"  Interictal: {total_stats['n_interictal']}")
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Build preprocessing cache")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/chbmit_raw",
        help="Path to raw CHB-MIT data",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/chbmit_cache",
        help="Path for output cache",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_run.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Specific subjects to process ('all' for auto-discover, default: from config)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing cache, don't rebuild",
    )
    parser.add_argument(
        "--cache_format",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Cache format: v1 (HDF5 per-subject) or v2 (memmap, RAM-stable)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Data type for v2 cache",
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    
    # Load config
    cfg = load_config(args.config)
    
    # Verify-only mode
    if args.verify_only:
        console.print("[bold blue]Verifying Cache Channel Consistency[/bold blue]")
        if args.cache_format == "v2":
            # For v2, check metadata.json
            if (out_root / "metadata.json").exists():
                import json
                with open(out_root / "metadata.json") as f:
                    meta = json.load(f)
                console.print(f"  [green]Cache V2 verified: {meta['n_samples']} samples[/green]")
                return 0
            else:
                console.print("[red]No v2 cache found[/red]")
                return 1
        else:
            verify_cache_channel_consistency(out_root)
            return 0
    
    # Get subjects
    if args.subjects and args.subjects[0] == "all":
        subjects = get_subject_list(data_root)
        console.print(f"[bold]Auto-discovered {len(subjects)} subjects[/bold]")
    elif args.subjects:
        subjects = args.subjects
    elif cfg.data.subjects:
        subjects = list(cfg.data.subjects)
    else:
        subjects = get_subject_list(data_root)
    
    console.print(f"[bold blue]Building Cache ({args.cache_format.upper()}) with Canonical Channels[/bold blue]")
    console.print(f"Data root: {data_root}")
    console.print(f"Output: {out_root}")
    console.print(f"Config: {args.config}")
    console.print(f"Format: {args.cache_format} (dtype: {args.dtype})")
    console.print(f"[bold]CANONICAL_CHANNELS: {len(CANONICAL_CHANNELS)} channels enforced[/bold]")
    console.print(f"Subjects: {', '.join(subjects[:10])}{'...' if len(subjects) > 10 else ''}")
    
    # Cache V2 mode
    if args.cache_format == "v2":
        stats = build_cache_v2(subjects, data_root, out_root, cfg, args.dtype)
        console.print("\n[bold green]Cache V2 build complete![/bold green]")
        return 0
    
    # V1 mode (original HDF5 per-subject)
    # Channel log file
    channel_log_dir = Path("reports/tables")
    channel_log_dir.mkdir(parents=True, exist_ok=True)
    channel_log_file = channel_log_dir / "channel_log.csv"
    
    # Clear log file
    with open(channel_log_file, "w", encoding="utf-8") as f:
        f.write("file,n_channels,channels\n")
    
    # Process each subject
    total_stats = {"n_windows": 0, "n_preictal": 0, "n_interictal": 0}
    all_channel_stats = {"total_files": 0, "files_with_missing": 0, "missing_channels": set()}
    
    for subject in subjects:
        stats = process_subject(subject, data_root, out_root, cfg, channel_log_file)
        for k in ["n_windows", "n_preictal", "n_interictal"]:
            total_stats[k] += stats.get(k, 0)
        
        if stats.get("channel_info"):
            cinfo = stats["channel_info"]
            all_channel_stats["total_files"] += cinfo.get("total_files", 0)
            all_channel_stats["files_with_missing"] += cinfo.get("files_with_missing", 0)
            all_channel_stats["missing_channels"].update(cinfo.get("missing_channels", set()))
    
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total windows:   {total_stats['n_windows']}")
    console.print(f"  Preictal:        {total_stats['n_preictal']}")
    console.print(f"  Interictal:      {total_stats['n_interictal']}")
    
    if total_stats['n_preictal'] > 0:
        ratio = total_stats['n_interictal'] / total_stats['n_preictal']
        console.print(f"  Class ratio:     {ratio:.1f}:1")
    
    # Channel consistency report
    console.print(f"\n[bold]Channel Consistency Report:[/bold]")
    console.print(f"  Total EDF files: {all_channel_stats['total_files']}")
    console.print(f"  Files with missing channels: {all_channel_stats['files_with_missing']}")
    if all_channel_stats['missing_channels']:
        console.print(f"  [yellow]Missing channels (filled with zeros): {sorted(all_channel_stats['missing_channels'])}[/yellow]")
    
    # Final verification
    try:
        verify_cache_channel_consistency(out_root)
    except AssertionError as e:
        console.print(f"[bold red]Channel consistency verification FAILED: {e}[/bold red]")
        return 1
    
    console.print(f"\n[dim]Channel log saved to: {channel_log_file}[/dim]")
    console.print("\n[bold green]Cache build complete with verified channel consistency![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
