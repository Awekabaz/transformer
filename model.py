import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 20000, d_model: int = 512):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model  # dimension of the embedding
        self.vocab_size = vocab_size  # vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.srt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of a shape (max_len, d_model)
        pe = torch.zeros(self.max_len, self.d_model)

        # Vector of shape (max_len, 1)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(
            1
        )  # numerator
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )  # denominator

        # sin to even indices in the vector
        pe[:, 0::2] = torch.sin(position * div_term)

        # cos to odd indices in the vector
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension (1, max_len, d_model)

        self.register_buffer(
            "pe", pe
        )  # Register as buffer to avoid being a parameter and save state

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        x = x + (self.pe[:, : x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps  # small value to avoid division by zero
        self.aplha = nn.Parameter(
            torch.ones(1)
        )  # learnable scale parameter, multiplication
        self.beta = nn.Parameter(torch.zeros(1))  # learnable shift parameter, addition

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        mean = nn.mean(x, dim=-1, keepdim=True)  # mean across the last dimension
        std = nn.std(
            x, dim=-1, keepdim=True
        )  # standard deviation across the last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # first linear layer
        self.linear2 = nn.Linear(d_ff, d_model)  # second linear layer
        self.dropout = nn.Dropout(dropout)  # dropout layer
        self.relu = torch.relu()

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model) ->  Linear 1 -> (batch_size, seq_len, d_ff)
        # -> ReLU -> (batch_size, seq_len, d_ff) -> Dropout
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
