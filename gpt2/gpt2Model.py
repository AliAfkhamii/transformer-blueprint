import torch
from torch import nn

from decoder_block import DecoderBlock


class GPT2Model(nn.Module):

    def __init__(self, vocab_size, model_dim, ff_dim, pe_max_len, num_heads,
                 dec_dropout=0.1, attn_dropout=0.1, emb_dropout=0.1, bias=False, num_layers=12):
        super().__init__()

        # learnt Token Embeddings
        self.wte = nn.Embedding(vocab_size, model_dim)
        # learnt Positional Embedding
        self.wpe = nn.Embedding(pe_max_len, model_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(num_heads, model_dim, ff_dim, dec_dropout, attn_dropout, bias)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # Language Model Head
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeddings = self.wte(input_ids)

        # Positional embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_embeddings = self.wpe(position_ids)

        x = self.emb_dropout(token_embeddings + position_embeddings)

        self.attention_weights = []
        for layer in self.layers:
            x = layer(x)
            self.attention_weights.append(layer.attention.attention_weights)

        x = self.ln_f(x)

        return self.lm_head(x)
