from datasets import load_dataset
import os
import sys
import sentencepiece as spm
from transformers import T5Tokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NamuwikiDataset:
    def __init__(self, vocab_path, max_seq_len=1024, batch_size=8):
        self.dataset = load_dataset("heegyu/namuwiki-extracted", split='train').select_columns('text')
        self.max_seq_len = max_seq_len
        if os.path.exists(vocab_path):
            self.tokenizer = T5Tokenizer(vocab_path)
        else:
            sys.stderr.write(f"Tokenizer model not found at {vocab_path}\n")
            sys.stderr.write("Please check the path and try again\n")
            sys.exit(1)

        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        self.pad = self.tokenizer.pad_token
        self.unk = self.tokenizer.unk_token
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token

        # self.dataset.set_format(type='torch')
        # self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)


    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        # batch: list of tensors
        bsz = len(batch)
        batch = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in batch]
        max_prompt_len = max(len(prompt) for prompt in batch)
        max_len = min(max_prompt_len, self.max_seq_len - 1)
        src = torch.full((bsz, max_len + 1), self.pad_id, dtype=torch.long, device=device)
        tgt = torch.full((bsz, max_len + 1), self.pad_id, dtype=torch.long, device=device)
        for i, prompt in enumerate(batch):
            prompt_len = min(len(prompt), max_len)
            src[i, 1 : 1 + prompt_len] = torch.tensor(prompt[:prompt_len], dtype=torch.long, device=device)
            src[i, 0] = self.bos_id

            tgt[i, :prompt_len] = torch.tensor(prompt[:prompt_len], dtype=torch.long, device=device)
            tgt[i, prompt_len] = self.eos_id

        return src, tgt

    def encode(self, text, eos=True, bos=False):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if eos:
            ids.append(self.eos_id)
        if bos:
            ids.insert(0, self.bos_id)
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)



if __name__ == '__main__':
    print("dataset test")
    dataset = NamuwikiDataset('NamuwikiTokenizer.model')

    print(dataset.eos_id, dataset.eos)
    print(dataset.bos_id, dataset.bos)
    print(dataset.pad_id, dataset.pad)
    print(dataset.unk_id, dataset.unk)


    texts = ["안녕하세요. 반갑습니다용",
             "ㅎㅇ"]
    ids = dataset.collate_fn(texts)
    print(ids)

    # print(dataset)