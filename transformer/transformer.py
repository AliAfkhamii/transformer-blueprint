from torch import nn

from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):

    def __init__(self,
                 enc_vocab_size, dec_vocab_size, embed_dim, model_dim, num_hiddens, enc_ff_dim, dec_ff_dim, num_heads,
                 pe_max_len=1000, enc_dropout=0.1, dec_dropout=0.1, bias=False, enc_num_layers=6, dec_num_layers=6
                 ):
        super().__init__()

        self.encoder = TransformerEncoder(enc_vocab_size, embed_dim, model_dim, num_hiddens, enc_ff_dim, num_heads,
                                          pe_max_len=pe_max_len, dropout=enc_dropout, bias=bias,
                                          num_layers=enc_num_layers)
        self.decoder = TransformerDecoder(dec_vocab_size, embed_dim, model_dim, num_hiddens, dec_ff_dim, num_heads,
                                          pe_max_len=pe_max_len, dropout=dec_dropout,
                                          bias=bias, num_layers=dec_num_layers)


    def forward(self, src, src_valid_lens, tgt, tgt_valid_lens):
        src = self.encoder(src, src_valid_lens)
        return self.decoder(tgt, src, tgt_valid_lens)
