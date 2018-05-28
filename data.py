#!/usr/bin/python3

""" Loads and preprocesses the dataset. """

import os

from vocab import Vocab

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
    sentences = []
    with open(path, "r") as f:
        for paragraph in f:
            for sentence in paragraph.split(" . "):
                tokens = sentence.split()
                if not tokens:
                    continue
                sentences.append([index.add(t.lower()) for t in tokens])

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
