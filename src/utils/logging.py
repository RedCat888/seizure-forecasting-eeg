"""Logging utilities with rich formatting."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.logging import RichHandler
from rich.console import Console

# Global console for rich output
console = Console()

# Loggers cache
_loggers = {}


def setup_logging(
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging with rich console output and optional file logging.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        log_file: Optional log filename (defaults to timestamp)
        
    Returns:
        Root logger
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)
    
    # File handler if log_dir specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"train_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class MetricsLogger:
    """Logger for training metrics to CSV."""
    
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False
        
    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to CSV file."""
        if step is not None:
            metrics = {"step": step, **metrics}
            
        # Write header on first call
        if not self._header_written:
            with open(self.log_path, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self._header_written = True
        
        # Append metrics
        with open(self.log_path, "a") as f:
            values = [str(v) for v in metrics.values()]
            f.write(",".join(values) + "\n")


def log_config(cfg, logger: logging.Logger) -> None:
    """Log configuration to logger."""
    from omegaconf import OmegaConf
    logger.info("Configuration:")
    for line in OmegaConf.to_yaml(cfg).split("\n"):
        if line.strip():
            logger.info(f"  {line}")
