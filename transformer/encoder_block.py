from torch import nn

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork


class AddNorm(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.Layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


class EncoderBlock(nn.Module):

    def __init__(self, num_hiddens, ff_dim, num_heads, d_model, dropout=0.1, bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(num_hiddens, num_heads, d_model, dropout, bias)
        self.ffn = FeedForwardNetwork(d_model, ff_dim)

    def forward(self, x):

        # attention sublayer followed by residual connection and Layer normalization
        y = self.add_norm(x, self.attention(x))

        # feed-forward sublayer followed by residual connection and Layer normalization
        return self.add_norm(y + self.ffn(y))
