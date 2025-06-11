from torch import nn

from ffn import FeedForwardNetwork
from transformer.attention import MultiHeadAttention


class DecoderBlock(nn.Module):

    def __init__(self, num_heads, model_dim, ff_dim, dec_dropout=0.1, att_dropout=0.1, bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, model_dim, att_dropout, bias)
        self.att_norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.dec_dropout = nn.Dropout(dec_dropout)

    def forward(self, x):
        norm_x = self.att_norm(x)
        x = x + self.dec_dropout(self.attention(
            norm_x, norm_x, norm_x, casual_mask=True)
        )

        return x + self.dec_dropout(
            self.ffn(self.ffn_norm(x))
        )
