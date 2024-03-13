import argparse

import sentencepiece as spm

def main(args):
    input = args.input
    model_prefix = args.model_prefix
    vocab_size = args.vocab_size
    model_type = args.model_type
    character_coverage = args.character_coverage
    spm.SentencePieceTrainer.train(
        f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size}" +
        f" --model_type={model_type} --character_coverage={character_coverage}" +
        " --max_sentence_length=8192" +  # 문장 최대 길이 (너무 길면 에러발생)
        " --pad_id=0 --pad_piece=<pad>" +  # pad (0)
        " --unk_id=1 --unk_piece=<unk>" +  # unknown (1)
        " --bos_id=2 --bos_piece=<s>" +  # begin of sequence (2)
        " --eos_id=3 --eos_piece=</s>" +  # end of sequence (3)
        " --user_defined_symbols=<sep>,<cls>,<mask>")  # 사용자 정의 토큰

def parseArgs():
    # By default, SentencePiece uses Unknown (<unk>), BOS (<s>) and EOS (</s>) tokens
    # which have the ids of 0, 1, and 2 respectively.
    args = argparse.ArgumentParser()
    args.add_argument('--input', type=str, help='input file')
    args.add_argument('--model_prefix', type=str, help='model prefix', default='tokenizer')
    args.add_argument('--vocab_size', type=int, help='vocab size', default=32000)
    args.add_argument('--model_type', type=str, help='model type', default='bpe')
    args.add_argument('--character_coverage', type=float, help='character coverage', default=1.0)
    return args.parse_args()

if __name__ == '__main__':
    args = parseArgs()
    main(args)
