import torch
from torch import nn


class MultiHeadAttentionLegacy(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 num_heads=8,
                 dropout=0.0,
                 bias=True,
                 kdim=None,
                 vdim=None,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.head_dim = embed_dim // num_heads
        if kdim is None:
            kdim = embed_dim
        if vdim is None:
            vdim = embed_dim

        self.kdim = kdim
        self.vdim = vdim

        # Linear layers for the query, key, and value for each head
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        # Get the batch size
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        # Project the query, key, and value
        q = self.q_proj(query)  # (bsz, tgt_len, embed_dim)
        k = self.k_proj(key)  # (bsz, src_len, embed_dim)
        v = self.v_proj(value)  # (bsz, src_len, embed_dim)

        # Reshape the query, key, and value to have the same shape as the number of heads
        q = q.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.contiguous().view(bsz * self.num_heads, src_len, self.head_dim)
        v = v.contiguous().view(bsz * self.num_heads, src_len, self.head_dim)

        # compute the attention scores
        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_softmax = nn.functional.softmax(attn_output_weights, dim=-1)

        # Apply causal mask
        if is_causal:
            attn_softmax = attn_softmax.tril()
        # Apply the attention mask
        if attn_mask is not None:
            attn_softmax = attn_mask * attn_softmax

        # Apply the dropout
        attn_softmax = self.dropout(attn_softmax) if self.dropout is not None else attn_softmax

        # Apply the attention to the value
        attn_output = torch.bmm(attn_softmax, v)

        # Reshape the attention output to have the same shape as the number of heads
        attn_output = attn_output.contiguous().view(bsz, tgt_len, embed_dim)  # it's the same as concatenating the heads

        # Apply the output projection
        attn_output = self.out_proj(attn_output)

        return attn_output
