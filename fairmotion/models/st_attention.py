
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoderLayer
from torch.nn.init import xavier_uniform_

from fairmotion.models import decoders
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm(src)


class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        nhead=1,
        dropout=0.1,
        device="cpu",
        src_length=120
    ):
        super(SpatioTemporalAttention, self).__init__()
        self.spatial_embedding = nn.Linear(input_dim, hidden_dim)
        self.temporal_idx = torch.as_tensor([range(src_length)])
        self.temporal_embedding = nn.Embedding(src_length, hidden_dim)
        self.temporal_attn1 = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=2*hidden_dim, dropout=dropout)
        self.temporal_attn2 = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=2*hidden_dim, dropout=dropout)
        self.spatial_attn1 = TransformerEncoderLayer(src_length, nhead, dim_feedforward=2*hidden_dim, dropout=dropout)
        self.spatial_attn2 = TransformerEncoderLayer(src_length, nhead, dim_feedforward=2*hidden_dim, dropout=dropout)
        self.ff = FeedForwardLayer(hidden_dim, dim_feedforward=1024, dropout=dropout)
        self.project_to_output = nn.Linear(hidden_dim, input_dim)

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):

        if max_len is None:
            max_len = tgt.shape[1]

        frame = self.temporal_embedding(self.temporal_idx.to(src.device))
        pose = self.spatial_embedding(src)
        x = pose + frame

        tx = x.transpose(0,1)
        sx = tx.transpose(0,2)

        tx = self.temporal_attn1(tx)
        tx = self.temporal_attn2(tx)

        sx = self.spatial_attn1(sx)
        sx = self.spatial_attn2(sx)

        x = tx + sx.transpose(0,2)
        x = self.ff(x)
        x = self.project_to_output(x.transpose(0,1))
        return x[:, :max_len, :]

