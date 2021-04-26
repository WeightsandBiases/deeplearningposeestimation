
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from torch.nn.init import xavier_uniform_

import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_layers=1,
        dropout=0.1,
        device="cpu",
        src_length=120
    ):
        super(Attention, self).__init__()
        self.embed_idx = torch.as_tensor([range(src_length)])
        self.pose_embedding = nn.Linear(input_dim, hidden_dim)
        self.frame_embedding = nn.Embedding(src_length, hidden_dim)
        self.attention1 = TransformerEncoderLayer(hidden_dim, 8)
        self.attention2 = TransformerEncoderLayer(hidden_dim, 8)
        self.project_to_output = nn.Linear(hidden_dim, input_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):

        src = src.transpose(0, 1)

        frame = self.frame_embedding(self.embed_idx.to(src.device)).transpose(0,1)
        pose = self.pose_embedding(src)
        x = pose + frame
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.project_to_output(x).transpose(0,1)
        return x[:, :24, :]

