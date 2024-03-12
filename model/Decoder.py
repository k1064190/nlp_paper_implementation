from torch import nn

from block.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    def __init__(self,
                 max_seq_len=512,
                 embed_dim=768,
                 num_heads=8,
                 num_layer=6,
                 use_legacy=False,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads,
                                                  use_legacy=use_legacy) for _ in range(num_layer)])

    def forward(self, q, encoder_out, tgt_mask=None, src_tgt_mask=None, is_causal=True):
        for layer in self.layers:
            q = layer(q, encoder_out, tgt_mask=tgt_mask, src_tgt_mask=src_tgt_mask, is_causal=is_causal)
        return q
