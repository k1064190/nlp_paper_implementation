import torch
from torch import nn

from embedding.PositionEncoding import PositionalEncoding

from embedding.TokenEmbedding import TokenEmbedding
from model.Encoder import Encoder
from model.Decoder import Decoder
from einops import rearrange, repeat


class Transformer(nn.Module):
    def __init__(self,
                 max_seq_len=512,
                 embed_dim=768,
                 num_heads=8,
                 num_layer=6,
                 vocab_size=32000,
                 use_legacy=False,
                 ):
        super().__init__()
        self.encoder = Encoder(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads, num_layer=num_layer,
                               use_legacy=use_legacy)
        self.decoder = Decoder(max_seq_len=max_seq_len, embed_dim=embed_dim, num_heads=num_heads, num_layer=num_layer,
                               use_legacy=use_legacy)

        self.te = TokenEmbedding(embed_dim, vocab_size)
        self.pe = PositionalEncoding(embed_dim, max_seq_len)

        self.generator = nn.Linear(embed_dim, vocab_size)

    def encode(self, src, src_mask=None):
        src = self.te(src)
        src = self.pe(src)
        return self.encoder(src, src_mask=src_mask, is_causal=False)

    def decode(self, tgt, encoder_out, tgt_mask=None, tgt_src_mask=None):
        tgt = self.te(tgt)
        tgt = self.pe(tgt)
        return self.decoder(tgt, encoder_out, tgt_mask=tgt_mask, tgt_src_mask=tgt_src_mask, is_causal=True)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src, src)
        tgt_mask = self.make_pad_mask(tgt, tgt)
        tgt_src_mask = self.make_pad_mask(tgt, src)
        encoder_out = self.encode(src)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask=tgt_mask, tgt_src_mask=tgt_src_mask)
        out = self.generator(tgt)
        out = nn.functional.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_pad_mask(self, q, kv, pad_idx=0):
        # q: [batch, q_len]
        # kv: [batch, kv_len]
        q_len = q.size(1)
        kv_len = kv.size(1)
        q_mask = q.ne(pad_idx)
        q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
        q_mask = repeat(q_mask, 'b 1 i 1 -> b 1 i k', k=kv_len)

        kv_mask = kv.ne(pad_idx)
        kv_mask = rearrange(kv_mask, 'b i -> b 1 1 i')
        kv_mask = repeat(kv_mask, 'b 1 1 i -> b 1 j i', j=q_len)

        mask = q_mask & kv_mask
        mask.requires_grad = False
        return mask
