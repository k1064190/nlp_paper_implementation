import argparse

import torch
import torch.nn as nn
from data import NamuwikiDataset
from model.DecoderOnlyTransformer import DecoderOnlyTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model,
             dataset: NamuwikiDataset,
             input_text,
             max_gen_length=100):
    """
    Generate text using the model and tokenizer
    :param model:
    :param dataset:
    :param input_text: list[str]
    :param max_gen_length:
    :return:
    """
    max_prompt_len = dataset.max_seq_len // 2 + 1 # 1 for bos token

    # tokenizer encodes batch of text to ids with padding of max length of the batch
    # input_text should not be beyond max_seq_len
    input_ids = [[dataset.bos_id] + dataset.tokenizer.encode(text, add_special_tokens=False) for text in input_text]
    max_prompt_len = max([len(ids) for ids in input_ids])
    min_prompt_len = min([len(ids) for ids in input_ids])
    full_len = min(dataset.max_seq_len, max_prompt_len + max_gen_length)
    assert max_prompt_len <= max_prompt_len, f"input text length should be less than {max_prompt_len}"

    # add padding to max_seq_len to make input_ids have the same length
    token = torch.full((len(input_ids), full_len), dataset.pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids):
        token[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    # generate text
    is_eos = torch.zeros(len(input_ids), dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        for i in range(min_prompt_len, full_len):
            out, _ = model(token[:, :i])
            next_token = out[:, -1].argmax(dim=-1) # [bsz] with the highest probability of vocab
            # if token[i] has pad token, replace it with next_token else leave it as it is
            w = token[:, i]
            token[:, i] = torch.where(w == dataset.pad_id, next_token, w)

            # if all tokens are eos token, break
            is_eos = is_eos | (token[:, i] == dataset.eos_id)
            if is_eos.all():
                break

    # decode token to text
    generated_text = [dataset.decode(ids) for ids in token]
    print("is_eos: ", is_eos)
    return generated_text


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--vocab_path', '-v', type=str, help='vocab model path', default='NamuwikiTokenizer.model')
    args.add_argument('--checkpoint_path', '-c', type=str, help='checkpoint path', default='model.pt')
    args.add_argument('--max_gen_length', '-m', type=int, help='max generation length', default=100)
    args.add_argument('--input_text_path', '-i', type=str, help='input text path', default='input.txt')

    args = args.parse_args()

    checkpoint_path = args.checkpoint_path
    max_gen_length = args.max_gen_length
    input_text_path = args.input_text_path
    vocab_path = args.vocab_path

    dataset = NamuwikiDataset(vocab_path)
    print("dataset loaded")
    model = DecoderOnlyTransformer(vocab_size=32107, use_legacy=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print("model loaded")

    with open(args.input_text_path, 'r', encoding='utf-8') as f:
        input_text = f.readlines()
    input_text = [text.strip() for text in input_text]
    print("input text loaded")

    generated_text = generate(model, dataset, input_text, max_gen_length)
    for text in generated_text:
        print(text)


