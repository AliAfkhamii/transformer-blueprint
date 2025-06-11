from torch import nn

class AddNorm(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))
