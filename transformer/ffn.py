from torch import nn


class FeedForwardNetwork(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super().__init__()

        # save hyperparameters
        self.model_dim = model_dim
        self.ff_dim = ff_dim

        # Linear Transformations
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)

        self.relu = nn.ReLU()

        # Explicit Xavier (Glorot) initialization. Note that PyTorch applies this by default
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        assert self.model_dim == x.shape[-1], f"the input last dimension must match the model last dimension {self.model_dim}"

        return self.fc2(
            self.relu(
                self.fc1(x)
            )
        )
