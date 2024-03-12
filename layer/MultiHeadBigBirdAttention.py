import torch
from torch import nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import BlockSparseAttention


class MultiHeadBigBirdAttention(nn.Module):
    def __init__(self,
                 block_size=32,
                 embed_dim=1024,
                 num_heads=8,
                 sequence_length=2048,
                 dropout=0.1,
                 ):
        super().__init__()

        BLOCKS = sequence_length // block_size
        causal_layout = torch.ones((num_heads, BLOCKS, BLOCKS))

        block_attn = BlockSparseAttention(layout=causal_layout, block_size=block_size, dropout=dropout,
                                          num_heads=num_heads)

        self.attn = MultiHeadDispatch(
            dim_model=embed_dim,
            residual_dropout=dropout,
            num_heads=num_heads,
            attention=block_attn
        )

    def forward(
            self,
            q, k, v,
            attn_mask=None,
            is_causal=False
    ):
        if is_causal:
            causal_mask = torch.tril(torch.ones((q.size(1), q.size(1))).to(q.device))
            attn_mask = attn_mask * causal_mask if attn_mask is not None else causal_mask
        return self.attn(q, k, v, attn_mask=attn_mask)
