import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any


def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal mask for the given sequence length.
    The mask is used to prevent attending to future tokens in the sequence.
    """
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_language: str,
        tgt_language: str,
        src_tokenizer,
        tgt_tokenizer,
        seq_len: int,
    ):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair["translation"][self.src_language]
        tgt_text = src_target_pair["translation"][self.tgt_language]

        encoder_input_tokens = self.src_tokenizer.encode(src_text).ids
        decoder_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        num_pad_tokens_encoder = self.seq_len - len(encoder_input_tokens) - 2
        num_pad_tokens_decoder = self.seq_len - len(decoder_input_tokens) - 1

        if num_pad_tokens_encoder < 0 or num_pad_tokens_decoder < 0:
            raise ValueError("Sequence length is too long.")

        # Add SOS and EOS tokens to the encoder input
        # and pad the sequence to the required length
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * num_pad_tokens_encoder, dtype=torch.int64
                ),
            ]
        )

        # Add SOS token to the decoder input
        # and pad the sequence to the required length
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * num_pad_tokens_decoder, dtype=torch.int64
                ),
            ]
        )

        # Create the label tensor with EOS token and padding
        # The label is the decoder input tokens with EOS at the end
        # and padded to the required length
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * num_pad_tokens_decoder, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len, "Encoder input length mismatch."
        assert decoder_input.size(0) == self.seq_len, "Decoder input length mismatch."
        assert label.size(0) == self.seq_len, "Label length mismatch."

        encoder_mask = encoder_input != self.pad_token
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0).int()

        decoder_mask = decoder_input != self.pad_token
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0).int() & causal_mask(
            len(decoder_input)
        )

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,  # (1 - batch, 1 - d_model, seq_len)
            "decoder_mask": decoder_mask,  # (1 - batch, 1 - d_model, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
