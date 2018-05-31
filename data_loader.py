#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Loads and preprocesses the "Wikitext long term dependency
language modeling dataset:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

Lowercases all the words and splits into sentences, omitting
the ending period. Add sentence start/end tokens.
"""

import os

from vocab import Vocab

SENT_START = "<sentence_start>"
SENT_END = "<sentence_end>"

def path(part):
    """ Gets the dataset for 'part' being train|test|valid. """
    assert part in ("train", "test", "valid")
    return os.path.join("wikitext-2", "wiki." + part + ".tokens")


def load(path, index):
    """ Loads the wikitext2 data at the given path using
    the given index (maps tokens to indices). Returns
    a list of sentences where each is a list of token
    indices.
    """
    start = index.add(SENT_START)
    sentences = []
    with open(path, "r") as f:
        for paragraph in f:
            for sentence in paragraph.split(" . "):
                tokens = sentence.split()
                if not tokens:
                    continue
                sentence = [index.add(SENT_START)]
                sentence.extend(index.add(t.lower()) for t in tokens)
                sentence.append(index.add(SENT_END))
                sentences.append(sentence)

    return sentences


def main():
    print("WikiText2 preprocessing test and dataset statistics")
    index = Vocab()
    for part in ("train", "valid", "test"):
        print("Processing", part)
        sentences = load(path(part), index)
        print("Found", sum(len(s) for s in sentences),
              "tokens in", len(sentences), "sentences")
    print("Found in total", len(index), "tokens")


if __name__ == '__main__':
    main()
