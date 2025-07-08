import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from typing import Dict, Any
from pathlib import Path

from dataset import BilingualDataset, causal_mask
from utils import set_seed

HG_DATASET_NAME = "Helsinki-NLP/opus_books"


def get_all_sentences(dataset, language: str):
    """
    Extract all sentences from the dataset for the specified language.
    """
    for item in dataset:
        yield item["translation"][language]


def build_tokenzier(config: Dict, dataset, language: str):
    tokenizer_path = Path(config["tokenizer_path"].format(language))

    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {language}... ({tokenizer_path})")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )

        tokenizer.train_from_iterator(
            get_all_sentences(dataset, language), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading existing tokenizer for {language}... ({tokenizer_path})")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def load_dataset_and_tokenizer(config: Dict):
    """
    Load the dataset and tokenizer for the specified language.
    """
    # Set seed for reproducible data splits
    if 'seed' in config:
        set_seed(config['seed'])

    def calcualate_max_seq_len(dataset, tokenizer, language: str) -> int:
        return max(
            len(tokenizer.encode(item["translation"][language]).ids) for item in dataset
        )

    dataset = load_dataset(
        HG_DATASET_NAME,
        f"{config['language_src']}-{config['language_tgt']}",
        split="train",
    )

    # Build or load the tokenizer for the specified languag
    tokenizer_src = build_tokenzier(config, dataset, config["language_src"])
    tokenizer_tgt = build_tokenzier(config, dataset, config["language_tgt"])

    train_size = int(config["train_size"] * len(dataset))
    val_size = int(config["val_size"] * len(dataset))
    test_size = len(dataset) - 520  # train_size - val_size
    train_set_raw, val_set_raw, test_set_raw, _ = random_split(
        dataset, (500, 20, 20, len(dataset) - 540),
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    train_set = BilingualDataset(
        train_set_raw,
        config["language_src"],
        config["language_tgt"],
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
    )

    val_set = BilingualDataset(
        val_set_raw,
        config["language_src"],
        config["language_tgt"],
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
    )

    test_set = BilingualDataset(
        test_set_raw,
        config["language_src"],
        config["language_tgt"],
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
    )

    print(
        f"Max sequence length for {config['language_src']}: {calcualate_max_seq_len(train_set_raw, tokenizer_src, config['language_src'])}"
    )
    print(
        f"Max sequence length for {config['language_tgt']}: {calcualate_max_seq_len(train_set_raw, tokenizer_tgt, config['language_tgt'])}"
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    )
