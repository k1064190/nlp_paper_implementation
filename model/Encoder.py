from torch import nn

from block.EncoderBlock import EncoderBlock


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
