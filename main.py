#!/usr/bin/python3

"""
TODO document
"""

from argparse import ArgumentParser
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
from vocab import Vocab
from log_timer import LogTimer


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        print(self)

    def forward(self, sents, sent_lens):
        embedded = self.embedding(sents)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, sent_lens)
        packed_out, _ = self.gru(packed_in)
        return F.log_softmax(self.fc1(packed_out.data), dim=1)


def train(data, model, optimizer, args, device):

    def batches():
        current = 0
        random.shuffle(data)
        while len(data) >= current + args.batch_size:
            sentences = data[current:current + args.batch_size]
            sentences.sort(key=lambda l: len(l), reverse=True)
            padded = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(s) for s in sentences])
            lens = torch.LongTensor([len(s) for s in sentences])
            yield padded.to(device), lens.to(device)
            current += args.batch_size

    log_timer = LogTimer(2)
    model.train()
    for epoch_ind in range(args.epochs):
        model.zero_grad()
        for batch_ind, (padded_sents, sent_lens) in enumerate(batches()):
            out = model(padded_sents, sent_lens)
            target = nn.utils.rnn.pack_padded_sequence(padded_sents, sent_lens)
            loss = F.nll_loss(out, target.data)
            loss.backward()
            optimizer.step()
            if log_timer():
                logging.info("Epoch %d/%d batch %d loss %.2f",
                             epoch_ind, args.epochs, batch_ind, loss.item())


def parse_args():
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--epochs", type=int, default=10)
    argp.add_argument("--batch-size", type=int, default=32)

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")

    vocab = Vocab()
    train_data = data.load(data.path("train"), vocab)
    model = Net(len(vocab), 128, 128, 1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    train(train_data, model, optimizer, args, device)


if __name__ == '__main__':
    main()
