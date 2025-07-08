import torch
import numpy as np
import random
from typing import Dict


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): Random seed value
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility")


def configure_reproducibility(config: Dict):
    """
    Configure reproducibility settings based on config.

    Args:
        config (Dict): Configuration dictionary containing 'seed' key
    """
    set_seed(config["seed"])
