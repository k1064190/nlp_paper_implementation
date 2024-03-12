import math
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, embed_dim, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim


    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.embed_dim)
        return out