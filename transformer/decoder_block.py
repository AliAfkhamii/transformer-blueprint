from torch import nn

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from addnorm import AddNorm


class DecoderBlock(nn.Module):

    def __init__(self, ff_dim, num_heads, model_dim, self_att_dropout=0.1, cross_att_dropout=0.1, msa_norm_dropout=0.1,
                 mca_norm_dropout=0.1, ff_norm_dropout=0.1, bias=False):
        super().__init__()

        self.self_attention = MultiHeadAttention(num_heads, model_dim, dropout=self_att_dropout, bias=bias)
        self.cross_attention = MultiHeadAttention(num_heads, model_dim, dropout=cross_att_dropout, bias=bias)
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)
        self.add_norm1 = AddNorm(model_dim, msa_norm_dropout)
        self.add_norm2 = AddNorm(model_dim, mca_norm_dropout)
        self.add_norm3 = AddNorm(model_dim, ff_norm_dropout)

    def forward(self, x, x_enc, valid_len):
        # masked Multi-Head Self Attention sublayer
        x = self.add_norm1(x, self.self_attention(queries=x, keys=x, values=x, casual_mask=True))

        # Multi-Head Cross Attention sublayer
        x = self.add_norm2(x, self.cross_attention(queries=x, keys=x_enc, values=x_enc, valid_len=valid_len))

        # Position-Wise Feed-Forward Network sublayer
        x = self.add_norm3(x, self.ffn(x))

        return x
