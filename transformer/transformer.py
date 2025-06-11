from torch import nn

from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):

    def __init__(self,
                 enc_vocab_size, dec_vocab_size, embed_dim, model_dim, enc_ff_dim, dec_ff_dim, num_heads,
                 pe_max_len=1000, pe_dropout=0.1, att_dropout=0.1, att_norm_dropout=0.1, ff_norm_dropout=0.1,
                 bias=False, enc_num_layers=6, dec_num_layers=6
                 ):
        super().__init__()

        self.encoder = TransformerEncoder(enc_vocab_size, embed_dim, model_dim, enc_ff_dim, num_heads, pe_max_len,
                                          pe_dropout, att_dropout, att_norm_dropout, ff_norm_dropout, bias,
                                          enc_num_layers)
        self.decoder = TransformerDecoder(dec_vocab_size, embed_dim, model_dim, dec_ff_dim, num_heads,
                                          pe_max_len=pe_max_len, self_att_dropout=0.1, cross_att_dropout=0.1,
                                          pe_dropout=0.1, msa_norm_dropout=0.1, mca_norm_dropout=0.1,
                                          ff_norm_dropout=0.1, bias=bias, num_layers=dec_num_layers)

    def forward(self, src, src_valid_lens, tgt, tgt_valid_lens):
        src = self.encoder(src, src_valid_lens)
        return self.decoder(tgt, src, tgt_valid_lens)
