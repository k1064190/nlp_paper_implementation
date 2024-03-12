import torch
from torch import nn

class ResidualConnection(nn.Module):
    def __init__(self, embed_dim=768, pre_norm=True, dropout=0.1, norm="LayerNorm"):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout = nn.Dropout(dropout)
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm == "BatchNorm":
            self.norm = nn.BatchNorm1d(embed_dim)
        else:
            raise ValueError(f"Unknown norm type {norm}")

    def forward(self, x, sublayer):
        out = x
        if self.pre_norm:
            x = self.norm(x)
            x = self.dropout(x)
            x = sublayer(x)
        else:
            x = sublayer(x)
            x = self.norm(x)
            x = self.dropout(x)
        return out + x