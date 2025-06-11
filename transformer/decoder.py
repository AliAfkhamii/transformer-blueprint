from torch import nn

from positional_encoding import SinusoidalPE
from decoder_block import DecoderBlock
from embedding import Embedding


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, model_dim, ff_dim, num_heads, pe_max_len=1000,
                 self_att_dropout=0.1, cross_att_dropout=0.1, pe_dropout=0.1, msa_norm_dropout=0.1,
                 mca_norm_dropout=0.1, ff_norm_dropout=0.1, bias=False,
                 num_layers=6):
        super().__init__()

        self.embedding = Embedding(vocab_size, embed_dim, model_dim)
        self.pe = SinusoidalPE(model_dim, pe_max_len, pe_dropout)
        self.linear = nn.Linear(model_dim, vocab_size)

        self.layers = nn.ModuleList([
            DecoderBlock(ff_dim, num_heads, model_dim, self_att_dropout, cross_att_dropout, msa_norm_dropout,
                 mca_norm_dropout, ff_norm_dropout, bias)
            for _ in range(num_layers)
        ])

    def forward(self, x, x_enc, valid_lens):
        # scaled embedding and adding sinusoidal positional encoding pattern
        x = self.pe(self.embedding(x))

        self.attention_weights = []
        for i, layer in enumerate(self.layers):
            x = layer(x, x_enc, valid_lens)
            self.attention_weights.append({
                "self": layer.self_attention.attention_weights,
                "cross": layer.cross_attention.attention_weights
            })

        return self.linear(x)
