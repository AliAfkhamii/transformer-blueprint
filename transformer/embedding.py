import torch
from torch import nn

import math


class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_dim, model_dim):
        super().__init__()

        # save hyperparameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim

        # construct layers
        self.proj = nn.Linear(in_features=embed_dim, out_features=model_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim)

        # set scaling factor
        self.scaling_factor = float(math.sqrt(model_dim))

    def forward(self, x):
        # x is in shape (Batch_size, sequence_length); no representative dimension  for each token
        # batch_size, seq_length = x.size()

        # apply token embedding [shape(Batch_size, sequence_length, embedding_dimension)]
        token_embedding = self.embedding(x)

        # project the embedding dimension to model dimension(space) and apply scaling
        model_embedding = self.proj(token_embedding) * self.scaling_factor

        # return model embedding in shape (Batch-size, sequence_length, model_dimension)
        return model_embedding
