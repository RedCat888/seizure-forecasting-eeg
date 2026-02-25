#!/usr/bin/env python3
"""
Run all experiments and generate summary results.

Executes:
1. Label sanity visualization
2. Cache build with channel verification
3. Within-subject experiments
4. Cross-subject experiments (small and medium)
5. Evaluation with alarm metrics
6. Summary results table

Usage:
    python scripts/run_experiments.py --mode full
    python scripts/run_experiments.py --mode quick  # Just runs small experiments
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table


console = Console()


def run_command(cmd: str, description: str, check: bool = True) -> bool:
    """Run a command and print status."""
    console.print(f"\n[bold cyan]>>> {description}[/bold cyan]")
    console.print(f"[dim]{cmd}[/dim]")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=False,
        )
        if result.returncode == 0:
            console.print(f"[green]✓ {description} completed[/green]")
            return True
        else:
            console.print(f"[red]✗ {description} failed (exit code: {result.returncode})[/red]")
            return False
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ {description} failed: {e}[/red]")
        return False


def collect_results(runs_dir: Path) -> pd.DataFrame:
    """Collect results from all run directories."""
    results = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Try to find metrics
        metrics_file = run_dir / "eval_results" / "eval_metrics.csv"
        if not metrics_file.exists():
            metrics_file = run_dir / "final_metrics.txt"
        
        if not metrics_file.exists():
            continue
        
        run_name = run_dir.name
        
        # Parse split mode and subjects from run name
        split_mode = "cross_subject"
        subject = "multiple"
        
        if "within_subject" in run_name:
            split_mode = "within_subject"
            # Extract subject from name like deep_within_subject_chb01_xxx
            parts = run_name.split("_")
            for i, p in enumerate(parts):
                if p.startswith("chb"):
                    subject = p
                    break
        
        # Read metrics
        metrics = {"split_mode": split_mode, "subject": subject, "run_name": run_name}
        
        if metrics_file.suffix == ".csv":
            try:
                df = pd.read_csv(metrics_file)
                if len(df) > 0:
                    for col in df.columns:
                        if col not in ["split_mode", "subject"]:
                            metrics[col] = df[col].iloc[0]
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read {metrics_file}: {e}[/yellow]")
        else:
            # Parse text file
            try:
                with open(metrics_file, "r") as f:
                    for line in f:
                        if ":" in line:
                            key, val = line.split(":", 1)
                            key = key.strip().lower().replace(" ", "_")
                            try:
                                metrics[key] = float(val.strip())
                            except ValueError:
                                pass
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read {metrics_file}: {e}[/yellow]")
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def create_summary_table(results_df: pd.DataFrame, output_path: Path):
    """Create and save summary results table."""
    # Ensure required columns exist
    required_cols = ["split_mode", "subject", "auroc", "auprc", "sensitivity", "approx_fah"]
    for col in required_cols:
        if col not in results_df.columns:
            results_df[col] = 0.0 if col not in ["split_mode", "subject"] else ""
    
    # Select and rename columns for output
    summary_df = results_df[[
        "split_mode", "subject", "auroc", "auprc", "sensitivity", "approx_fah"
    ]].copy()
    
    summary_df.columns = [
        "split_mode", "subjects", "AUROC", "AUPRC", "sensitivity", "FAH"
    ]
    
    # Add mean warning time placeholder (would need to compute from eval)
    summary_df["mean_warning_time"] = "N/A"
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    console.print(f"\n[bold]Summary saved to: {output_path}[/bold]")
    
    # Print table
    table = Table(title="Experiment Results Summary")
    table.add_column("Split Mode", style="cyan")
    table.add_column("Subjects", style="white")
    table.add_column("AUROC", style="green")
    table.add_column("AUPRC", style="green")
    table.add_column("Sensitivity", style="yellow")
    table.add_column("FAH", style="red")
    
    for _, row in summary_df.iterrows():
        table.add_row(
            str(row["split_mode"]),
            str(row["subjects"]),
            f"{row['AUROC']:.4f}" if isinstance(row['AUROC'], float) else str(row['AUROC']),
            f"{row['AUPRC']:.4f}" if isinstance(row['AUPRC'], float) else str(row['AUPRC']),
            f"{row['sensitivity']:.4f}" if isinstance(row['sensitivity'], float) else str(row['sensitivity']),
            f"{row['FAH']:.2f}" if isinstance(row['FAH'], float) else str(row['FAH']),
        )
    
    console.print(table)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Experiment mode: quick (small subset) or full (all subjects)",
    )
    parser.add_argument(
        "--skip_cache",
        action="store_true",
        help="Skip cache building (use existing cache)",
    )
    parser.add_argument(
        "--within_subjects",
        type=str,
        nargs="+",
        default=["chb01", "chb02"],
        help="Subjects for within-subject experiments",
    )
    args = parser.parse_args()
    
    console.print("[bold blue]=" * 60 + "[/bold blue]")
    console.print("[bold blue]SEIZURE FORECASTING EXPERIMENTS[/bold blue]")
    console.print("[bold blue]=" * 60 + "[/bold blue]")
    console.print(f"Mode: {args.mode}")
    console.print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    runs_dir = project_root / "runs"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Track success/failure
    steps = []
    
    # =========================================================================
    # STEP 1: Label Sanity Visualization
    # =========================================================================
    success = run_command(
        "python scripts/label_sanity.py --config configs/small_run.yaml",
        "Label Sanity Visualization"
    )
    steps.append(("Label Sanity", success))
    
    # =========================================================================
    # STEP 2: Build Cache with Channel Verification
    # =========================================================================
    if not args.skip_cache:
        config = "configs/small_run.yaml" if args.mode == "quick" else "configs/medium_run.yaml"
        success = run_command(
            f"python scripts/build_cache.py --config {config}",
            "Build Cache with Channel Verification"
        )
        steps.append(("Cache Build", success))
        
        # Verify cache
        run_command(
            "python scripts/build_cache.py --verify_only",
            "Verify Cache Channel Consistency",
            check=False
        )
    
    # =========================================================================
    # STEP 3: Within-Subject Experiments
    # =========================================================================
    for subject in args.within_subjects:
        success = run_command(
            f"python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject {subject}",
            f"Within-Subject Training ({subject})"
        )
        steps.append((f"Within-Subject {subject}", success))
    
    # =========================================================================
    # STEP 4: Cross-Subject Experiments
    # =========================================================================
    config = "configs/small_run.yaml" if args.mode == "quick" else "configs/medium_run.yaml"
    success = run_command(
        f"python scripts/train_deep.py --config {config} --split_mode cross_subject",
        "Cross-Subject Training"
    )
    steps.append(("Cross-Subject", success))
    
    # =========================================================================
    # STEP 5: Evaluation
    # =========================================================================
    console.print("\n[bold cyan]>>> Finding and evaluating checkpoints...[/bold cyan]")
    
    # Find latest run directories and evaluate
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            
            checkpoint = run_dir / "checkpoints" / "best.pt"
            if checkpoint.exists():
                eval_dir = run_dir / "eval_results"
                
                # Determine split mode from run name
                split_mode_arg = ""
                subject_arg = ""
                if "within_subject" in run_dir.name:
                    split_mode_arg = "--split_mode within_subject"
                    # Extract subject
                    parts = run_dir.name.split("_")
                    for p in parts:
                        if p.startswith("chb"):
                            subject_arg = f"--subject {p}"
                            break
                else:
                    split_mode_arg = "--split_mode cross_subject"
                
                run_command(
                    f"python scripts/full_eval.py --checkpoint {checkpoint} --config {config} {split_mode_arg} {subject_arg}",
                    f"Evaluate {run_dir.name}",
                    check=False
                )
    
    # =========================================================================
    # STEP 6: Collect Results and Create Summary
    # =========================================================================
    console.print("\n[bold cyan]>>> Collecting results...[/bold cyan]")
    
    results_df = collect_results(runs_dir)
    
    if len(results_df) > 0:
        summary_path = tables_dir / "summary_results.csv"
        create_summary_table(results_df, summary_path)
    else:
        console.print("[yellow]No results found to summarize[/yellow]")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    console.print("\n[bold blue]=" * 60 + "[/bold blue]")
    console.print("[bold blue]EXPERIMENT SUMMARY[/bold blue]")
    console.print("[bold blue]=" * 60 + "[/bold blue]")
    
    for step_name, success in steps:
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status} {step_name}")
    
    n_success = sum(1 for _, s in steps if s)
    n_total = len(steps)
    console.print(f"\nCompleted: {n_success}/{n_total} steps")
    
    if n_success == n_total:
        console.print("\n[bold green]All experiments completed successfully![/bold green]")
    else:
        console.print("\n[bold yellow]Some experiments failed. Check logs above.[/bold yellow]")
    
    return 0 if n_success == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
