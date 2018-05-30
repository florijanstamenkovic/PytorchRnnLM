#!/usr/bin/python3

"""
TODO document
"""

from argparse import ArgumentParser
import logging
import math
import random
import sys

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

        # Custom init for the embedding as the output uses the same (tied)
        # weights.
        emb_w = torch.Tensor(vocab_size, embedding_dim)
        stdv = 1. / math.sqrt(emb_w.size(1))
        emb_w.uniform_(-stdv, stdv)
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      _weight=emb_w)

        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers)

        # Output bias, the weights are tied to the embedding.
        self.out_b = torch.nn.Parameter(torch.zeros(vocab_size))

        print(self)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.embedding(packed_sents.data), packed_sents.batch_sizes)
        # GRU output is (packed_sequence_out, hidden_state)
        out = self.gru(embedded_sents)[0].data
        # Do weight sharing between embedding and output weights.
        out = out.mm(self.embedding.weight.t()) + self.out_b

        return F.log_softmax(out, dim=1)


def train(data, model, optimizer, args, device, vocab):

    def batches():
        current = 0
        random.shuffle(data)
        while len(data) >= current + args.batch_size:
            sentences = data[current:current + args.batch_size]
            sentences.sort(key=lambda l: len(l), reverse=True)
            yield [torch.LongTensor(s) for s in sentences]
            current += args.batch_size

    log_timer = LogTimer(2)
    model.train()
    for epoch_ind in range(args.epochs):
        model.zero_grad()
        for batch_ind, sents in enumerate(batches()):
            x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
            y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
            if device.type == 'cuda':
                x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = F.nll_loss(out, y.data)
            if log_timer() or batch_ind == 0:
                # Calculate and log perplexity, a common metric for language
                # models.
                prob = out.exp()[
                    torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
                perplexity = 2 ** prob.log2().neg().mean().item()
                logging.info("Epoch %d/%d, batch %d, loss %.3f, perplexity %.2f",
                             epoch_ind, args.epochs, batch_ind, loss.item(),
                             perplexity)

            loss.backward()
            optimizer.step()


def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--embedding-dim", type=int, default=384)
    argp.add_argument("--gru-hidden", type=int, default=384)
    argp.add_argument("--gru-layers", type=int, default=1)

    argp.add_argument("--epochs", type=int, default=10)
    argp.add_argument("--batch-size", type=int, default=48)
    argp.add_argument("--lr", type=float, default=0.0003)

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")

    vocab = Vocab()
    train_data = data.load(data.path("train"), vocab)
    model = Net(len(vocab), args.embedding_dim,
                args.gru_hidden, args.gru_layers).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    train(train_data, model, optimizer, args, device, vocab)


if __name__ == '__main__':
    main()
