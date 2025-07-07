from pathlib import Path
from typing import Dict


def get_config() -> Dict:
    config = {
        "batch_size": 6,
        "num_epochs": 5,
        "learning_rate": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "train_size": 0.9,
        "language_src": "en",
        "language_tgt": "it",
        "model_folder": "weights",
        "model_name": "transformer_model",
        "preload_weights": None,
        "tokenizer_path": "tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/transformer",
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
