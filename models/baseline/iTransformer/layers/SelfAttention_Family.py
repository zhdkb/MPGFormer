import torch
import torch.nn as nn
import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch, query_len, heads, dim = queries.shape
        _, _, _, value_dim = values.shape
        scale = self.scale or 1.0 / sqrt(dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch, query_len, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        heads = self.n_heads

        queries = self.query_projection(queries).view(batch, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, heads, -1)
        values = self.value_projection(values).view(batch, key_len, heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(batch, query_len, -1)

        return self.out_projection(out), attn
