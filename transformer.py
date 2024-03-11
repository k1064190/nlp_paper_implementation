import torch
from torch import nn
import transformers
from torch._C._return_types import topk
from transformers import BertModel, BertTokenizer, BertConfig
from datasets import load_dataset
from xformers.components import MultiHeadDispatch
from xformers.components.attention import BlockSparseAttention
import math
from transformers import AutoModel, AutoTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Create a matrix of the position
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)  # (bsz, seq_len, embed_dim)

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


class EncoderBlock(nn.Module):
    def __init__(self,
                 max_seq_len=512,
                 embed_dim=768,
                 num_heads=8,
                 use_legacy=False,
                 ):
        super().__init__()
        # self.attn = MultiHeadAttentionLegacy(embed_dim=embed_dim, num_heads=num_heads) if use_legacy else nn.MultiheadAttention(embed_dim, num_heads)
        self.attn = (MultiHeadBigBirdAttention
                     (embed_dim=embed_dim, num_heads=num_heads, sequence_length=max_seq_len, dropout=0.1)) \
            if not use_legacy else MultiHeadAttentionLegacy(embed_dim=embed_dim, num_heads=num_heads)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        res = q
        # prenorm
        q = self.ln1(q)
        # attention
        q = self.attn(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        q = self.dropout(q)
        # residual
        q = res + q

        res = q
        # prenorm
        q = self.ln2(q)
        # feedforward
        q = self.ffn(q)
        # residual
        q = res + q
        q = self.dropout(q)

        return q


class Encoder(nn.Module):
    def __init__(self,
                 max_seq_len=512,
                 embed_dim=768,
                 num_heads=8,
                 num_layer=6,
                 use_legacy=False,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads,
                                                  use_legacy=use_legacy) for _ in range(num_layer)])

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        for layer in self.layers:
            q = layer(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return q


class Transformer(nn.Module):
    def __init__(self,
                 max_seq_len=1024,
                 embed_dim=768,
                 num_heads=8,
                 num_layer=6,
                 use_legacy=False,
                 ):
        super().__init__()
        self.encoder = Encoder(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads, num_layer=num_layer,
                               use_legacy=use_legacy)
        self.decoder = Encoder(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads, num_layer=num_layer,
                               use_legacy=use_legacy)

        self.pe = PositionalEncoding(embed_dim, max_seq_len)

    def src_mask(self, src):



    def forward(self, src, tgt):


