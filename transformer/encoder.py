from torch import nn

from positional_encoding import SinusoidalPE
from encoder_block import EncoderBlock
from embedding import Embedding


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, model_dim, num_hiddens, ff_dim, num_heads, pe_max_len=1000, dropout=0.1,
                 bias=False, num_layers=6):
        super().__init__()

        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, embed_dim, model_dim)
        self.PE = SinusoidalPE(num_hiddens, dropout=dropout, max_len=pe_max_len)

        # construct num_layer encoder layers
        self.layers = nn.ModuleList([
            EncoderBlock(num_hiddens, ff_dim, num_heads, model_dim, dropout, bias)
            for _ in range(num_layers)
        ])

    def forward(self, x, valid_lens):
        x = self.PE(self.embedding(x))

        self.attention_weights = []
        for i, layer in enumerate(self.layers):
            x = layer(x, valid_lens)
            self.attention_weights.append(layer.attention.attention_weights)

        return x
