import torch
from torch import nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.ff1 = nn.Linear(embed_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.ff2(self.dropout(self.activation(self.ff1(x))))