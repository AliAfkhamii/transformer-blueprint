import torch
from torch import nn

import math


class SinusoidalPE(nn.Module):

    def __init__(self, num_hidden, max_len=1000, dropout=0.1):
        super().__init__()

        # save hyperparameters
        self.num_hidden = num_hidden
        self.max_len = max_len

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # create a long positional embedding
        self.pe = torch.zeros((1, self.max_len, self.num_hidden), dtype=torch.float32)
        self.initialize_pe()

    def initialize_pe(self):
        # positional encoding trigonometric function input [i/10000^(2j/d)]
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(dim=1)
        j_2 = torch.arange(0, self.num_hidden, step=2)
        div_term = torch.exp(- math.log(10000) * (j_2 / self.num_hidden))

        self.pe[:, :, 0::2] = torch.sin(position * div_term)
        self.pe[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1], :].to(x.device))
