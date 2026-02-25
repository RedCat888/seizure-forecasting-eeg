#!/usr/bin/env python3
"""
Generate cache report with statistics.

Reports:
- Total windows
- Preictal percentage
- Cache size on disk
- Estimated RAM usage during training

Usage:
    python scripts/cache_report.py --cache_dir data/chbmit_cache_v2
"""

import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def get_directory_size(path: Path) -> int:
    """Get total size of all files in directory."""
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def report_v2_cache(cache_dir: Path, output_file: Path):
    """Generate report for Cache V2."""
    metadata_path = cache_dir / "metadata.json"
    
    if not metadata_path.exists():
        console.print(f"[red]Error: No metadata.json found in {cache_dir}[/red]")
        return
    
    with open(metadata_path) as f:
        meta = json.load(f)
    
    # Calculate disk usage
    disk_size = get_directory_size(cache_dir)
    disk_size_gb = disk_size / (1024**3)
    
    # Estimate RAM usage
    n_samples = meta["n_samples"]
    n_channels = meta["n_channels"]
    window_len = meta["window_len"]
    n_features = meta["n_features"]
    dtype_size = 2 if meta["dtype"] == "float16" else 4
    
    # Per-sample memory footprint (during batch loading)
    sample_size_data = n_channels * window_len * 4  # float32 at runtime
    sample_size_features = n_features * 4  # float32
    sample_size_labels = 4 * 4  # 4 float32 values
    sample_size_total = sample_size_data + sample_size_features + sample_size_labels
    
    # Batch RAM (256 samples)
    batch_ram_mb = 256 * sample_size_total / (1024**2)
    
    # Display report
    console.print("\n[bold cyan]Cache V2 Report[/bold cyan]")
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Windows", f"{n_samples:,}")
    table.add_row("Preictal Windows", f"{meta['n_preictal']:,}")
    table.add_row("Interictal Windows", f"{meta['n_interictal']:,}")
    table.add_row("Preictal %", f"{meta['preictal_pct']:.2f}%")
    table.add_row("", "")
    table.add_row("Subjects", f"{len(meta['subjects'])}")
    table.add_row("Channels", f"{n_channels}")
    table.add_row("Window Length", f"{window_len} samples ({window_len/256:.1f}s @ 256Hz)")
    table.add_row("Features", f"{n_features}")
    table.add_row("Storage Dtype", meta["dtype"])
    table.add_row("", "")
    table.add_row("Disk Size", f"{disk_size_gb:.2f} GB")
    table.add_row("RAM per Batch (256)", f"{batch_ram_mb:.1f} MB")
    table.add_row("RAM for Full Load*", f"{n_samples * sample_size_total / (1024**3):.1f} GB")
    
    console.print(table)
    console.print("[dim]*V2 does NOT load full dataset into RAM - uses memmap![/dim]")
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("CHB-MIT Cache V2 Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Windows:     {n_samples:,}\n")
        f.write(f"Preictal Windows:  {meta['n_preictal']:,}\n")
        f.write(f"Interictal Windows:{meta['n_interictal']:,}\n")
        f.write(f"Preictal %:        {meta['preictal_pct']:.2f}%\n")
        f.write(f"\n")
        f.write(f"Subjects:          {len(meta['subjects'])}\n")
        f.write(f"Channels:          {n_channels}\n")
        f.write(f"Window Length:     {window_len} samples ({window_len/256:.1f}s @ 256Hz)\n")
        f.write(f"Features:          {n_features}\n")
        f.write(f"Storage Dtype:     {meta['dtype']}\n")
        f.write(f"\n")
        f.write(f"Disk Size:         {disk_size_gb:.2f} GB\n")
        f.write(f"RAM per Batch:     {batch_ram_mb:.1f} MB (batch_size=256)\n")
        f.write(f"Memmap Backed:     Yes (minimal RAM usage)\n")
        f.write(f"\n")
        f.write(f"Subjects: {', '.join(meta['subjects'])}\n")
    
    console.print(f"\n[dim]Report saved to: {output_file}[/dim]")


def report_v1_cache(cache_dir: Path, output_file: Path):
    """Generate report for Cache V1 (HDF5 per-subject)."""
    import h5py
    
    h5_files = list(cache_dir.glob("*.h5"))
    
    if not h5_files:
        console.print(f"[red]Error: No HDF5 files found in {cache_dir}[/red]")
        return
    
    total_windows = 0
    total_preictal = 0
    total_interictal = 0
    subjects = []
    
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            y_cls = f["y_cls"][:]
            n_windows = len(y_cls)
            n_preictal = int((y_cls == 1).sum())
            n_interictal = int((y_cls == 0).sum())
            
            total_windows += n_windows
            total_preictal += n_preictal
            total_interictal += n_interictal
            subjects.append(h5_path.stem)
    
    disk_size = get_directory_size(cache_dir)
    disk_size_gb = disk_size / (1024**3)
    
    preictal_pct = 100.0 * total_preictal / max(total_windows, 1)
    
    # RAM estimate (full load)
    # 18 channels, 7680 samples, float32 = 18 * 7680 * 4 = 552960 bytes per window
    ram_gb = total_windows * 552960 / (1024**3)
    
    console.print("\n[bold cyan]Cache V1 Report[/bold cyan]")
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Windows", f"{total_windows:,}")
    table.add_row("Preictal Windows", f"{total_preictal:,}")
    table.add_row("Interictal Windows", f"{total_interictal:,}")
    table.add_row("Preictal %", f"{preictal_pct:.2f}%")
    table.add_row("Subjects", f"{len(subjects)}")
    table.add_row("Disk Size", f"{disk_size_gb:.2f} GB")
    table.add_row("RAM for Full Load", f"{ram_gb:.1f} GB")
    
    console.print(table)
    console.print("[yellow]Warning: V1 cache loads ALL data into RAM![/yellow]")
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("CHB-MIT Cache V1 Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Windows:     {total_windows:,}\n")
        f.write(f"Preictal Windows:  {total_preictal:,}\n")
        f.write(f"Interictal Windows:{total_interictal:,}\n")
        f.write(f"Preictal %:        {preictal_pct:.2f}%\n")
        f.write(f"Subjects:          {len(subjects)}\n")
        f.write(f"Disk Size:         {disk_size_gb:.2f} GB\n")
        f.write(f"RAM for Full Load: {ram_gb:.1f} GB\n")
        f.write(f"\n")
        f.write(f"Subjects: {', '.join(subjects)}\n")
    
    console.print(f"\n[dim]Report saved to: {output_file}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Generate cache report")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/chbmit_cache_v2",
        help="Path to cache directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/tables/cache_report_full.txt",
        help="Output file path",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    output_file = Path(args.output)
    
    if not cache_dir.exists():
        console.print(f"[red]Error: Cache directory not found: {cache_dir}[/red]")
        return 1
    
    # Detect cache format
    if (cache_dir / "metadata.json").exists():
        report_v2_cache(cache_dir, output_file)
    elif list(cache_dir.glob("*.h5")):
        report_v1_cache(cache_dir, output_file)
    else:
        console.print(f"[red]Error: Unknown cache format in {cache_dir}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
