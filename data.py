from datasets import load_dataset
import os
import sys
import sentencepiece as spm
from transformers import AutoTokenizer
import torch

class NamuwikiDataset:
    def __init__(self, vocab_path, txt_path=None, max_seq_len=512, batch_size=8):
        self.dataset = load_dataset("heegyu/namuwiki-extracted", split='train')['text']
        self.max_seq_len = max_seq_len
        if os.path.exists(vocab_path):
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_path)
        else:
            if txt_path is None:
                txt_path = 'temp_txt.txt'
            if not os.path.exists(txt_path):
                self.write_to_text(txt_path)

            txt_file = open(txt_path, 'r', encoding='utf-8')
            vocab_path = vocab_path.split('.')[0]
            spm.SentencePieceTrainer.train(
                f"--input={input} --model_prefix={vocab_path} --vocab_size=32000" +
                f" --model_type=bpe --character_coverage=1.0" +
                " --max_sentence_length=999999" +  # 문장 최대 길이 (너무 길면 에러발생)
                " --pad_id=0 --pad_piece=<pad>" +  # pad (0)
                " --unk_id=1 --unk_piece=<unk>" +  # unknown (1)
                " --bos_id=2 --bos_piece=<s>" +  # begin of sequence (2)
                " --eos_id=3 --eos_piece=</s>" +  # end of sequence (3)
                " --user_defined_symbols=<sep>,<cls>,<mask>")  # 사용자 정의 토큰

            self.tokenizer = AutoTokenizer.from_pretrained(vocab_path + '.model')
            self.dataset = self.dataset.map(lambda e: self.tokenizer.encode('<s>' + e + '</s>', max_length=self.max_seq_len, padding='max_length', truncation=True), batched=True)
            self.dataset.set_format(type='torch')

            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def write_to_text(self, output_path="datasets"):
        with open(f"{output_path}/train.txt", "w", encoding='utf-8') as f:
            for example in self.dataset:
                f.write(example + "\n")

    def test_tokenizer(self):
        print(self.tokenizer.encode(self.dataset['train'][0]['text']))

    def __len__(self):
        return len(self.dataset['train'])

    def encode_batch(self, text_list):
        return self.tokenizer.batch_encode_plus(text_list, max_length=self.max_seq_len, pad_to_max_length=True, return_tensors='pt')


if __name__ == '__main__':
    print("dataset test")
    dataset = NamuwikiDataset('tokenizer.model')