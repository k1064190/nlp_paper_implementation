import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import DecoderOnlyTransformer
from tqdm import tqdm
from accelerate import Accelerator

from data import NamuwikiDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_ddp_accelerate(model, dataset, data_loader, optimizer, criterion, epochs, checkpoint_dir, args):
    accelerate = Accelerator()

    model, optimizer, data_loader = accelerate.prepare(model, optimizer, data_loader)

    model.train()
    for epoch in tqdm(range(epochs)):
        for batch_idx, src in enumerate(tqdm(data_loader)):
            src = src.to(device)
            # src is [bsz, :max_seq_len]
            # tgt is [bsz, 1:max_seq_len+1]
            tgt = src[:, 1:]
            src = src[:, :-1]
            optimizer.zero_grad()
            out, _ = model(src) # out is [bsz, max_seq_len, vocab_size]
            loss = criterion(out.reshape(-1, out.size(-1)), tgt.reshape(-1))  # this automatically ignores padding tokens
            accelerate.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"loss: {loss.item()}")

        if epoch % args.save_every == 0:
            accelerate.save(model.state_dict(), f"{checkpoint_dir}/model_{epoch}.pt")
            print(f"model saved at {checkpoint_dir}/model_{epoch}.pt")



def parseArgs():
    args = argparse.ArgumentParser()
    args.add_argument('--vocab_path', type=str, help='vocab model path', default='NamuwikiTokenizer.model')
    args.add_argument('--checkpoint_dir', type=str, help='checkpoint directory')    # not implemented yet
    args.add_argument('--batch_size', type=int, help='batch size', default=8)
    args.add_argument('--max_seq_len', type=int, help='max sequence length', default=1024)
    args.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    args.add_argument('--n_epochs', type=int, help='number of epochs', default=100)
    args.add_argument('--save_every', type=int, help='save checkpoint every n epochs', default=5)
    args.add_argument('--log_every', type=int, help='log every n steps', default=100)
    return args.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    dataset = NamuwikiDataset(args.vocab_path, args.max_seq_len, args.batch_size)
    dataloader = dataset.dataloader

    model = DecoderOnlyTransformer(vocab_size=32107, use_legacy=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)

    train_ddp_accelerate(model, dataset, dataloader, optimizer, criterion, args.n_epochs, args.checkpoint_dir, args)
