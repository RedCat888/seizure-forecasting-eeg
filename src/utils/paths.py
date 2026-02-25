"""Path management utilities."""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from omegaconf import DictConfig


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from this file to find project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent


def get_paths(cfg: Optional[DictConfig] = None) -> Dict[str, Path]:
    """
    Get common paths used throughout the project.
    
    Args:
        cfg: Optional config to use for paths
        
    Returns:
        Dict of path names to Path objects
    """
    root = get_project_root()
    
    paths = {
        "root": root,
        "data": root / "data",
        "raw_data": root / "data" / "chbmit_raw",
        "cache": root / "data" / "chbmit_cache",
        "runs": root / "runs",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "tables": root / "reports" / "tables",
        "configs": root / "configs",
    }
    
    # Override with config if provided
    if cfg is not None:
        if hasattr(cfg, "data"):
            if hasattr(cfg.data, "raw_root"):
                paths["raw_data"] = root / cfg.data.raw_root
            if hasattr(cfg.data, "cache_root"):
                paths["cache"] = root / cfg.data.cache_root
        if hasattr(cfg, "logging") and hasattr(cfg.logging, "run_dir"):
            paths["runs"] = root / cfg.logging.run_dir
    
    return paths


def get_run_dir(cfg: DictConfig, create: bool = True) -> Path:
    """
    Get or create a run directory for this experiment.
    
    Args:
        cfg: Config with logging settings
        create: Whether to create the directory
        
    Returns:
        Path to run directory
    """
    paths = get_paths(cfg)
    
    experiment_name = cfg.logging.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = paths["runs"] / f"{experiment_name}_{timestamp}"
    
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)
    
    return run_dir


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
