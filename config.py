from pathlib import Path
from typing import Dict


def get_config() -> Dict:
    config = {
        "batch_size": 12,
        "num_epochs": 3,
        "learning_rate": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "train_size": 0.8,
        "val_size": 0.1,
        "language_src": "en",
        "language_tgt": "it",
        "model_folder": "weights",
        "model_name": "transformer_model",
        "preload_weights": None,
        "tokenizer_path": "tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/transformer",
        "seed": 42,  # Random seed for reproducibility
    }
    return config


def get_weights_path(config, epoch: str) -> str:
    """
    Get the path to the model weights based on the configuration.
    """
    model_path = config["model_folder"]
    model_basename = config["model_name"]
    model_filename = f"{model_basename}_{epoch}.pt" if epoch else f"{model_basename}.pt"

    return str(Path(".") / model_path / model_filename)
