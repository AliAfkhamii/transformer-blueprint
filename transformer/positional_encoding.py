import torch
from torch import nn

import math


class SinusoidalPE(nn.Module):

    def __init__(self, model_dim, max_len=1000, dropout=0.1):
        super().__init__()

        # save hyperparameters
        self.model_dim = model_dim
        self.max_len = max_len

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # create a long positional embedding
        self.pe = torch.zeros((1, self.max_len, self.model_dim), dtype=torch.float32)
        self._initialize_pe()

    def _initialize_pe(self):
        # positional encoding trigonometric function input [i/10000^(2j/d)]
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(dim=1)
        j_2 = torch.arange(0, self.model_dim, step=2)
        div_term = torch.exp(- math.log(10000) * (j_2 / self.model_dim))

        self.pe[:, :, 0::2] = torch.sin(position * div_term)
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1], :].to(x.device))
