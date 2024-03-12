import argparse

from model.DecoderOnlyTransformer import DecoderOnlyTransformer
from model.Transformer import Transformer

def build_model(model, max_seq_len, embed_dim, num_heads, num_layer, vocab_size, use_legacy, model_type):
    if model == 'transformer':
        model = Transformer
    elif model == 'decoder_only_transformer':
        model = DecoderOnlyTransformer
    else:
        raise ValueError(f"Unknown model {model}")

    return model(max_seq_len, embed_dim, num_heads, num_layer, vocab_size, use_legacy)