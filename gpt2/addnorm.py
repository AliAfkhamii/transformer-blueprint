from torch import nn


class AddNorm(nn.Module):

    def __init__(self, dim, dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x, sublayer):

        return x + self.dropout()

