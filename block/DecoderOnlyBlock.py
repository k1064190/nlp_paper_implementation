import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from xformers import _is_triton_available
# if package triton is installed, import MultiHeadBigBirdAttention from layer.MultiHeadBigBirdAttention
if _is_triton_available():
    from layer.MultiHeadBigBirdAttention import MultiHeadBigBirdAttention
from layer.PositionWiseFeedForward import PositionWiseFeedForward
from layer.MultiHeadAttentionLegacy import MultiHeadAttentionLegacy
from layer.ResidualConnection import ResidualConnection


class DecoderOnlyBlock(nn.Module):
    def __init__(self,
                 max_seq_len=1024,
                 embed_dim=768,
                 num_heads=8,
                 use_legacy=False,
                 ):
        super().__init__()
        # self.attn = MultiHeadAttentionLegacy(embed_dim=embed_dim, num_heads=num_heads) if use_legacy else nn.MultiheadAttention(embed_dim, num_heads)
        if not use_legacy:
            if _is_triton_available():
                self.attn = (MultiHeadBigBirdAttention
                             (embed_dim=embed_dim, num_heads=num_heads, sequence_length=max_seq_len, dropout=0.1))
            else:
                print("Triton is not available, using legacy attention. Please install triton to use BigBird attention.")
                self.attn = MultiHeadAttentionLegacy(embed_dim=embed_dim, num_heads=num_heads)
        else:
            self.attn = MultiHeadAttentionLegacy(embed_dim=embed_dim, num_heads=num_heads)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = PositionWiseFeedForward(embed_dim=embed_dim)
        self.residual = ResidualConnection(embed_dim=embed_dim, pre_norm=True)

    def forward(self, q, tgt_mask=None, is_causal=True):
        q = self.residual(q, lambda x: self.attn(x, x, x, attn_mask=tgt_mask, is_causal=is_causal))
        q = self.residual(q, lambda x: self.ffn(x))
        return q
