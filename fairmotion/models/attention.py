
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

from fairmotion.models import decoders
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm(src)

class Attention(nn.Module):
    """RNN model for sequence prediction. The model uses a single RNN module to
    take an input pose, and generates a pose prediction for the next time step.

    Attributes:
        input_dim: Size of input vector for each time step
        hidden_dim: RNN hidden size
        num_layers: Number of layers of RNN cells
        dropout: Probability of an element to be zeroed
        device: Device on which to run the RNN module
    """

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
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src, tgt: Tensors of shape (batch_size, seq_len, input_dim)
            max_len: Maximum length of sequence to be generated during
                inference. Set None during training.
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        # convert src, tgt to (seq_len, batch_size, input_dim) format

        src = src.transpose(0, 1)

        frame = self.frame_embedding(self.embed_idx.to(src.device)).transpose(0,1)
        pose = self.pose_embedding(src)
        print(frame.shape, pose.shape)
        x = pose + frame
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.project_to_output(x).transpose(0,1)
        return x[:, :24, :]

