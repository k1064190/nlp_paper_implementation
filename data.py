from datasets import load_dataset
import os
import sys
import sentencepiece as spm
from transformers import AutoTokenizer

class NamuwikiDataset:
    def __init__(self, vocab_path, txt_path=None):
        self.dataset = load_dataset("heegyu/namuwiki-extracted")

        if os.path.exists(vocab_path):
            self.sp = AutoTokenizer.from_pretrained(vocab_path)
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

            self.sp = AutoTokenizer.from_pretrained(vocab_path)

    def write_to_text(self, output_path="datasets"):
        for split in self.dataset.keys():
            with open(f"{output_path}/{split}.txt", "w", encoding='utf-8') as f:
                for example in self.dataset[split]:
                    f.write(example["title"] + "\n")
                    f.write(example["text"] + "\n")

    def test_tokenizer(self):
        print(self.sp.encode(self.dataset['train'][0]['text']))