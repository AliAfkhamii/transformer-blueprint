from torch import nn

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from addnorm import AddNorm


class EncoderBlock(nn.Module):

    def __init__(self, num_hiddens, ff_dim, num_heads, d_model, dropout=0.1, bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(num_hiddens, num_heads, d_model, dropout, bias)
        self.ffn = FeedForwardNetwork(d_model, ff_dim)
        self.add_norm = AddNorm(d_model, dropout)

    def forward(self, x):
        # attention sublayer followed by residual connection and Layer normalization
        y = self.add_norm(x, self.attention(x))

        # feed-forward sublayer followed by residual connection and Layer normalization
        return self.add_norm(y + self.ffn(y))
