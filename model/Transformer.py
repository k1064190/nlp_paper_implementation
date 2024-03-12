from torch import nn

from embedding.PositionEncoding import PositionalEncoding
from model.Encoder import Encoder


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
