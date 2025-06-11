from torch import nn


class FeedForwardNetwork(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)

        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(
                self.gelu(
                    self.fc1(x))
            )
