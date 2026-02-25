"""
Configuration schema with defaults.
Ensures no missing-key crashes by providing sensible defaults.
"""

from omegaconf import OmegaConf, DictConfig


# Default configuration schema
CONFIG_DEFAULTS = {
    "seed": 42,
    "experiment_name": "seizure_forecasting",
    
    "data": {
        "data_root": "data/chbmit_raw",
        "cache_root": "data/chbmit_cache_v2",
        "cache_format": "v2",
        "subjects": None,
    },
    
    "signal": {
        "target_sfreq": 256,
        "lowcut": 0.5,
        "highcut": 50.0,
        "notch_freq": 60.0,
        "amp_uv_thresh": 500,
    },
    
    "windowing": {
        "window_sec": 30,
        "step_sec": 15,
        "preictal_min": 30,
        "gap_sec": 300,
        "postictal_min": 30,
        "interictal_buffer_min": 60,
        "tau_sec": 1800,
    },
    
    "features": {
        "compute_bandpower": True,
        "compute_ratios": True,
        "compute_line_length": True,
        "compute_spectral_entropy": True,
        "compute_hjorth": True,
        "compute_kurtosis": True,
        "nan_fill_value": 0.0,  # How to handle NaN in features
        "bands": {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 50],
        },
    },
    
    "spectrogram": {
        "n_fft": 256,
        "hop_length": 64,
        "win_length": 256,
        "log_offset": 1.0,
    },
    
    "split": {
        "mode": "cross_subject",
        "train_subjects": [],
        "val_subjects": [],
        "test_subjects": [],
        "within_train_ratio": 0.7,
        "seed": 42,
    },
    
    "model": {
        "type": "fusion_net",
        "use_features": True,
        "cnn_channels": [32, 64, 128],
        "cnn_kernel_sizes": [3, 3, 3],
        "mlp_hidden": [256, 128],
        "fusion_hidden": 128,
        "dropout": 0.3,
        "n_features": 30,
        "seq_len": 7680,
        "n_channels": 18,
    },
    
    "training": {
        "batch_size": 256,
        "epochs": 30,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "use_amp": True,
        "grad_clip": 1.0,
        "early_stopping_patience": 10,
        "gradient_accumulation_steps": 1,
        "scheduler": "cosine",
        "scheduler_warmup_epochs": 2,
        "lambda_soft": 0.5,
        "weight_by_proximity": True,
        "num_workers": 0,  # Safe default for Windows
        "pin_memory": True,
        "persistent_workers": False,
    },
    
    "loss": {
        "type": "focal",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "regression_weight": 0.1,
        "soft_weight": 0.0,
    },
    
    "augmentation": {
        "enabled": True,
        "gaussian_noise_std": 0.01,
        "time_shift_max_samples": 64,
        "amplitude_scale_range": [0.95, 1.05],
        "channel_dropout_prob": 0.05,
    },
    
    "alarm": {
        "smoothing_method": "ema",
        "smoothing_alpha": 0.2,
        "smoothing_window": 6,
        "persistence_k": 3,
        "use_hysteresis": True,
        "hysteresis_gap": 0.1,
        "refractory_sec": 1200,
    },
    
    "evaluation": {
        "fah_targets": [0.1, 0.2, 0.5, 1.0],
        "threshold_sweep_points": 100,
    },
    
    "logging": {
        "save_every_n_epochs": 5,
        "log_every_n_steps": 50,
        "save_best_only": True,
        "save_last": True,
        "run_dir": "runs",
        "experiment_name": "loso",
    },
    
    "output": {
        "run_dir": "runs",
        "save_best_by_auc": True,
        "save_best_by_fah": True,
        "save_training_curves": True,
        "save_manifest": True,
    },
}


def apply_defaults(cfg: DictConfig) -> DictConfig:
    """
    Apply default values to config, filling in any missing keys.
    
    Args:
        cfg: User-provided config (may have missing keys)
        
    Returns:
        Complete config with defaults filled in
    """
    defaults = OmegaConf.create(CONFIG_DEFAULTS)
    # Merge: user config takes precedence over defaults
    merged = OmegaConf.merge(defaults, cfg)
    return merged


def get_safe(cfg: DictConfig, key: str, default=None):
    """
    Safely get a nested config value with a default.
    
    Args:
        cfg: Config dict
        key: Dot-separated key path (e.g., "logging.save_every_n_epochs")
        default: Default value if key not found
        
    Returns:
        Value at key or default
    """
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default
