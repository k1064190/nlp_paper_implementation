from torch import nn

from layer.MultiHeadAttentionLegacy import MultiHeadAttentionLegacy
from layer.MultiHeadBigBirdAttention import MultiHeadBigBirdAttention


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
