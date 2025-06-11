from torch import nn

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from addnorm import AddNorm


class EncoderBlock(nn.Module):

    def __init__(self, ff_dim, num_heads, model_dim, att_dropout=0.1, att_norm_dropout=0.1, ff_norm_dropout=0.1,
                 bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, model_dim, att_dropout, bias)
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)
        self.add_norm1 = AddNorm(model_dim, att_norm_dropout)
        self.add_norm2 = AddNorm(model_dim, ff_norm_dropout)

    def forward(self, x, valid_len):
        # attention sublayer followed by residual connection and Layer normalization
        y = self.add_norm1(x, self.attention(queries=x, keys=x, values=x, valid_len=valid_len))

        # feed-forward sublayer followed by residual connection and Layer normalization
        return self.add_norm2(y + self.ffn(y))
