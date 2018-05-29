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

    def forward(self, packed_sents):
        # Calculate GRU outputs
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.embedding(packed_sents.data), packed_sents.batch_sizes)
        out = self.fc1(self.gru(embedded_sents)[0].data)

        sm = F.softmax(out, dim=1)
        probs = sm[torch.arange(0, len(packed_sents.data), dtype=torch.int64),
                   packed_sents.data]
        perplexity = 2 ** (probs.log2().mean().neg().item())

        # logging.info("Softmax shape: %r, mean %.3f, selected %.3f, "
        #              "perplexity %.3f, data %.3f",
        #           sm.shape, sm.mean().item(), probs.mean().item(), perplexity,
        #              packed_sents.data.sum().item())

        return F.log_softmax(out, dim=1), perplexity


def train(data, model, optimizer, args, device, vocab):

    def batches():
        current = 0
        random.shuffle(data)
        while len(data) >= current + args.batch_size:
            sentences = data[current:current + args.batch_size]
            sentences.sort(key=lambda l: len(l), reverse=True)
            packed = nn.utils.rnn.pack_sequence(
                [torch.LongTensor(s) for s in sentences])
            yield packed if device.type == 'cpu' else packed.cuda()
            current += args.batch_size

    log_timer = LogTimer(2)
    model.train()
    for epoch_ind in range(args.epochs):
        model.zero_grad()
        for batch_ind, packed_sents in enumerate(batches()):
            out, perplexity = model(packed_sents)
            loss = F.nll_loss(out, packed_sents.data)
            if log_timer():
                logging.info("Epoch %d/%d, batch %d, loss %.3f, perplexity %.2f",
                             epoch_ind, args.epochs, batch_ind, loss.item(),
                             perplexity)
                logging.info("Sentences")
                padded, _ = nn.utils.rnn.pad_packed_sequence(packed_sents, True)
                logging.info(" ".join([vocab[i.item()] for i in padded[0]]))
                for i in range(20):
                    pos = padded[0][i].item()
                    logging.info("\t%s - %.3f",
                                 vocab[pos],
                                 out[i, pos].item())
            loss.backward()
            optimizer.step()


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
    train(train_data, model, optimizer, args, device, vocab)


if __name__ == '__main__':
    main()
