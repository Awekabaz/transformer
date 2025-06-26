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


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension of each head

        assert (
            self.d_k * self.n_heads == self.d_model
        ), "d_model must be divisible by n_heads"
        # W_k: Key (seq_length, d_model) -> W_k (d_model, d_model) -> (seq_length, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        # W_q: Query (seq_length, d_model) -> W_q (d_model, d_model) -> (seq_length, d_model)
        self.w_q = nn.Linear(d_model, d_model)

        # W_v: Value (seq_length, d_model) -> W_v (d_model, d_model) -> (seq_length, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # W_o: Output (seq_length, h * d_v) -> W_o (h * d_v, d_model) -> (seq_length, d_model) where h * d_v = d_model
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)  # dropout layer

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout = None, mask=None):
        # query, key, value are of shape (batch_size, n_heads, seq_len, d_k)
        d_k = query.size(-1)  # dimension of the key/query

        # (batch_size, n_heads, seq_len, d_k) matmul (batch_size, n_heads, d_k, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # mask is of shape (batch_size, 1, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # softmax over the last dimension
        # (batch_size, n_heads, seq_len, seq_len)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_probs = dropout(attention_probs)  # apply dropout

        # (batch_size, n_heads, seq_len, d_k), attention_probs
        return (attention_scores @ value), attention_probs

    def forward(self, q, k, v, mask=None):

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # divide by the number of heads (split by embedding not sentence into h heads)
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k) -> transpose -> (batch_size, n_heads, seq_len, d_k)
        # intuition: each head will see whole sequence, but with smaller part of the embedding
        query = query.view(
            query.size(0), query.size(1), self.n_heads, self.d_k
        ).transpose(1, 2)

        key = key.view(key.size(0), key.size(1), self.n_heads, self.d_k).transpose(1, 2)

        value = value.view(
            value.size(0), value.size(1), self.n_heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, self.dropout, mask
        )

        # (batch_size, n_heads, seq_len, d_k) -> transpose (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, n_heads * d_k) where n_heads * d_k = d_model
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer):
        # x is of shape (batch_size, seq_len, d_model)
        # sublayer is a previous layer (e.g. MultiHeadAttentionBlock or FeedForwardBlock)
        return x + self.dropout(sublayer(self.layer_norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout), ResidualConnection(dropout)]
        )

    def forward(self, x, src_mask=None):
        # src_mask needed for masking the padding tokens in the input sequence

        # Apply self-attention block with residual connection
        # x is of shape (batch_size, seq_len, d_model)
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # Apply feed-forward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, src_mask=None):
        # x is of shape (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)  # (batch_size, seq_len, d_model)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(dropout),
                ResidualConnection(dropout),
                ResidualConnection(dropout),
            ]
        )

    def forward(self, x, encoder_output, src_mask=None, target_mask=None):
        # src_mask needed for masking the padding tokens in the input sequence
        # target_mask needed for masking the future tokens in the target sequence

        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )

        # x used as key from decoder, encoder_output used as value and query from encoder
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )

        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask=None, target_mask=None):
        # x is of shape (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self.norm(x)  # (batch_size, seq_len, d_model)


class LinearLayer(nn.Module):
    def __init__(self, d_model: int = 512, vocab_size: int = 20000):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(
            self.linear(x), dim=-1
        )  # (batch_size, seq_len, vocab_size)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbedding,
        tgt_embedding: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        linear_layer: LinearLayer,
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear_layer = linear_layer

    def encode(self, src, src_mask=None):
        # src is of shape (batch_size, seq_len)
        src = self.src_embedding(src)
        src = self.src_pos(src)  # (batch_size, seq_len, d_model)
        return self.encoder(src, src_mask)  # (batch_size, seq_len, d_model)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.src_embedding(tgt)
        tgt = self.src_pos(tgt)  # (batch_size, seq_len, d_model)

        return self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )  # (batch_size, seq_len, d_model)

    def linear(self, x):
        return self.linear_layer(x)


def build_transformer(
    src_vocab: int,
    tgt_vocab: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,  # dimension of the model
    N: int = 6,  # number of encoder/decoder blocks
    h: int = 8,  # number of heads in multi-head attention
    dropout=0.1,
    d_ff: int = 2048,  # dimension of the feed-forward network
) -> Transformer:
    # Create embedding layers
    src_embedding = InputEmbedding(src_vocab, d_model)
    tgt_embedding = InputEmbedding(tgt_vocab, d_model)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_layers = nn.ModuleList(
        [
            EncoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(N)
        ]
    )

    # Create decoder blocks
    decoder_layers = nn.ModuleList(
        [
            DecoderBlock(
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout,
            )
            for _ in range(N)
        ]
    )

    encoder = Encoder(encoder_layers)
    decoder = Decoder(decoder_layers)

    # Create linear layer
    linear_layer = LinearLayer(d_model, tgt_vocab)

    transformer = Transformer(
        encoder,
        decoder,
        src_embedding,
        tgt_embedding,
        src_pos,
        tgt_pos,
        linear_layer,
    )

    # Initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
