"""Reproducibility utilities for deterministic training."""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Make cuDNN deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker with unique seed for DataLoader multiprocessing.
    
    Use as: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
