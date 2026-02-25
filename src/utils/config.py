"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Optional, Any, Dict
from omegaconf import OmegaConf, DictConfig
import yaml


_CONFIG: Optional[DictConfig] = None


def load_config(config_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: Optional dict of overrides to apply
        
    Returns:
        OmegaConf DictConfig object
    """
    global _CONFIG
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Apply overrides if provided
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Store globally
    _CONFIG = cfg
    
    return cfg


def get_config() -> DictConfig:
    """Get the currently loaded configuration."""
    if _CONFIG is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _CONFIG


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf to regular dict."""
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs, later ones override earlier."""
    return OmegaConf.merge(*configs)
