import matplotlib.pyplot as plt
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
            # add bos token
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        else:
            sys.stderr.write(f"Tokenizer model not found at {vocab_path}\n")
            sys.stderr.write("Please check the path and try again\n")
            sys.exit(1)

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collate_fn)


    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        batch = [x['text'] for x in batch]
        # encode text to ids with max length of the batch
        src = [[self.bos_id] + self.tokenizer.encode(text, add_special_tokens=False) + [self.eos_id] for text in batch] # [bsz, each_text_len + 2]
        max_len = max([len(ids) for ids in src])
        max_len = min(max_len, self.max_seq_len + 1)
        # if id exceeds max_seq_len, truncate it and doesn't append eos token
        src = [ids[:max_len] if len(ids) > max_len else ids + [self.pad_id] * (max_len - len(ids)) for ids in src] # [bsz, max_len+1]
        # <s> + text + </s> + pad if text length is less than max_seq_len else <s> + text[:max_seq_len]
        # when inference, text length is less than max_seq_len, because we set max_prompt_len less than max_seq_len
        src = torch.tensor(src, dtype=torch.long, device=device)    # [bsz, max_len+1]

        return src

    def decode(self, id):
        return self.tokenizer.decode(id)



if __name__ == '__main__':
    print("dataset test")
    dataset = NamuwikiDataset('NamuwikiTokenizer.model')

    print(dataset.eos_id)
    print(dataset.bos_id)
    print(dataset.pad_id)
    print(dataset.unk_id)
    print(dataset.vocab_size)



    # import matplotlib.pyplot as plt
    #
    # def tokenize_dataset(examples):
    #     return dataset.tokenizer(examples['text'], add_special_tokens=False)
    # tokenized_dataset = dataset.dataset.map(tokenize_dataset, batched=True)
    #
    # # save tokenized dataset
    # tokenized_dataset.save_to_disk('tokenized_namuwiki')
    #
    # lengths = [len(x['input_ids']) for x in tokenized_dataset]
    #
    # plt.figure(figsize=(10, 5))
    # plt.hist(lengths, bin=20, alpha=0.7, color='blue')
    # plt.xlabel('Length of tokenized data')
    # plt.ylabel('Frequency')
    # plt.show()

    # # load dataloader
    # dataloader = iter(dataset.dataloader)
    #
    # # test next batch
    # batch = next(dataloader)
    # print(batch)
    #
    # # test decode
    # print([dataset.decode(x) for x in batch])

