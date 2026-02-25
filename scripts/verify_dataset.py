#!/usr/bin/env python3
"""
Verify CHB-MIT dataset integrity and generate summary report.

Usage:
    python scripts/verify_dataset.py --data_root data/chbmit_raw
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chbmit import (
    get_subject_list,
    get_edf_files,
    get_subject_info,
    build_dataset_index,
)


def main():
    parser = argparse.ArgumentParser(description="Verify CHB-MIT dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/chbmit_raw",
        help="Path to CHB-MIT data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for index",
    )
    args = parser.parse_args()
    
    console = Console()
    data_root = Path(args.data_root)
    
    console.print(f"\n[bold blue]CHB-MIT Dataset Verification[/bold blue]")
    console.print(f"Data root: {data_root.absolute()}\n")
    
    # Check if directory exists
    if not data_root.exists():
        console.print(f"[bold red]Error: Data root not found: {data_root}[/bold red]")
        console.print("\nPlease download the dataset first. See docs/DATA_DOWNLOAD.md")
        sys.exit(1)
    
    # Get subjects
    try:
        subjects = get_subject_list(data_root)
    except Exception as e:
        console.print(f"[bold red]Error reading subjects: {e}[/bold red]")
        sys.exit(1)
    
    if not subjects:
        console.print("[bold red]No subject folders found![/bold red]")
        console.print("Expected folders named 'chb01', 'chb02', etc.")
        sys.exit(1)
    
    console.print(f"Found [bold green]{len(subjects)}[/bold green] subjects\n")
    
    # Build detailed index
    console.print("Building dataset index...")
    index = build_dataset_index(data_root)
    
    # Create summary table
    table = Table(title="Dataset Summary by Subject")
    table.add_column("Subject", style="cyan")
    table.add_column("EDF Files", justify="right")
    table.add_column("Seizures", justify="right", style="yellow")
    table.add_column("Files w/ Seizures", justify="right")
    
    for subject in subjects:
        info = index.subject_info.get(subject)
        if info:
            table.add_row(
                subject,
                str(len(info.edf_files)),
                str(info.total_seizures),
                str(len(info.files_with_seizures)),
            )
    
    console.print(table)
    
    # Print totals
    console.print(f"\n[bold]Totals:[/bold]")
    console.print(f"  Total EDF files: [green]{index.total_edf_files}[/green]")
    console.print(f"  Total seizures:  [yellow]{index.total_seizures}[/yellow]")
    
    # Check for RECORDS files
    records_file = data_root / "RECORDS"
    records_seizures_file = data_root / "RECORDS-WITH-SEIZURES"
    
    console.print(f"\n[bold]Metadata Files:[/bold]")
    console.print(f"  RECORDS:              {'[green]Found[/green]' if records_file.exists() else '[red]Not found[/red]'}")
    console.print(f"  RECORDS-WITH-SEIZURES: {'[green]Found[/green]' if records_seizures_file.exists() else '[red]Not found[/red]'}")
    
    # Detailed seizure info
    console.print(f"\n[bold]Seizures per Subject:[/bold]")
    for subject in subjects:
        info = index.subject_info.get(subject)
        if info and info.total_seizures > 0:
            console.print(f"  {subject}: {info.total_seizures} seizures in {len(info.files_with_seizures)} files")
            for fname, seizures in info.seizures_by_file.items():
                if seizures:
                    times = ", ".join([f"{s}-{e}s" for s, e in seizures])
                    console.print(f"    {fname}: {times}")
    
    # Save index if requested
    if args.output:
        from src.chbmit.indexing import save_index
        output_path = Path(args.output)
        save_index(index, output_path)
        console.print(f"\n[green]Index saved to {output_path}[/green]")
    
    console.print("\n[bold green]Verification complete![/bold green]\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
