import torch
from torch import nn
import torch.nn.functional as F

import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, d_model=None, dropout=0.1, bias=False):
        super().__init__()

        # save hyperparameters
        self.num_heads = num_heads
        self.d_model = d_model or num_hiddens

        assert self.d_model % num_heads == 0, \
            "model dimension must be divisible by number of heads. " \
            f"got model_dimension {self.d_model} and num_heads {self.num_heads}"

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # projection matrices
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, mask=None):
        batch_size, seq_length, model_dim = queries.size()  # using "seq_length" and "no. of queries" interchangeably

        assert model_dim == self.d_model, \
            f"input dimension must match model dimension got {self.d_model} and {model_dim}"

        # project q, k, v
        # shape (batch_size, seq_length, num_hiddens) -> shape (batch_size * num_heads, seq_length, model_dim / num_heads)
        q = self.split_by_heads(self.W_q(queries))
        k = self.split_by_heads(self.W_k(keys))
        v = self.split_by_heads(self.W_v(values))

        # [softmax(qk^T / sqrt(d))] with masking the paddings
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        self.attention_weights = self.masked_softmax(scores, mask)

        # scaled dot product attention heads
        attention_heads = torch.bmm(
            self.dropout(self.attention_weights), v)

        return self.W_o(
            self.transpose_outputs(attention_heads)
        )

    def split_by_heads(self, x):
        """
        input shape (batch_size, seq_length, model_dim)
        output shape (batch_size * num_heads, seq_length, model_dim / num_heads)

        abbreviations:
            batch_size: B
            seq_length or no. of q, k_v : n
            model_dim: d_m
            num_heads: h

        shape transformations:
            (B, n, d_m) -> (B, n, h, d_m / h) -> (B, h, n, d_m / h) -> (B*h, n, d_m / h)
        """

        # -> (B, n, h, d_m / h)
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)

        # -> (B, h, n, d_m / h)
        x = x.permute(0, 2, 1, 3)

        # -> (B*h, n, d_m / h)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        return x

    def transpose_outputs(self, x):
        """
        it basically reverts the original shape it's already had.

        abbreviations:
            batch_size: B
            seq_length or no. of q, k_v : n
            model_dim: d_m
            num_heads: h
            key-value pair dimension: d_v

        from:
            shape(B*h, n, d_v)
        to:
            shape(B, n, d_m) or shape(B, n, h*dv)

        shape transformations:
            (B*h, n, dv) -> (B, h, n, dv) -> (B, n, h, dv) -> (B, n, h*dv)
        """

        # -> (B, h, n, dv)
        x = x.reshape(x.shape[0], self.num_heads, x.shape[1], -1)

        # -> (B, n, h, dv)
        x = x.transpose(1, 2)

        # -> (B, n, h*dv)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return x

    @staticmethod
    def masked_softmax(logits, valid_len=None, dim=-1, value=-10e6):

        if valid_len is None:
            return F.softmax(logits, dim=dim)

        shape = logits.size()

        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, shape[1])
        else:
            valid_len = valid_len.reshape(-1)

        # flatten logits from shape (B, no. of q, seq_len) into a 2D Tensor of shape(B * no. q, seq_len)
        logits = logits.reshape(-1, shape[-1])

        # indices of keys
        max_len = shape[-1]

        # indices where a row vector of query indices < valid len column vector is True
        mask = \
            torch.arange(max_len, dtype=torch.float, device=logits.device)[None, :] \
            < \
            valid_len[:, None]  # a column vector of valid lengths for each query, reshaped for broadcasting

        logits.masked_fill_(~mask, value)

        # reshape logits back to original shape and apply softmax along the intended axis
        return F.softmax(logits.reshape(shape), dim=dim)
