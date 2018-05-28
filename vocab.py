# !/usr/bin/env python3
# -*- coding: UTF-8 -*-


class Vocab(object):
    """ Converts word tokens to indices, and vice versa. """

    def __init__(self):
        super().__init__()

        self._tok_to_ind = {}
        self._ind_to_tok = []
        self._counts = []

    def count(self, word):
        """ Returns the count of times a word occured. """
        return self._counts[self[word]]

    def add(self, word):
        """ Adds the given word to the dict, or increases it's
        count if alreays present. Returns it's index.
        """
        ind = self._tok_to_ind.get(word, None)
        if ind is None:
            ind = len(self._ind_to_tok)
            self._ind_to_tok.append(word)
            self._tok_to_ind[word] = ind
            self._counts.append(1)
        else:
            self._counts[ind] += 1

        return ind

    def word_counts(self):
        """ Returns a list of tuples (word, count) sorted descending. """
        return sorted(zip(self._ind_to_tok, self._counts),
                      key=lambda t: t[1], reverse=True)

    def __len__(self):
        return len(self._tok_to_ind)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ind_to_tok[index]
        else:
            return self._tok_to_ind.get(word)

    def __setitem__(self, word, value):
        raise "Can't set items directly, use add(word) instead"

    def __delitem__(self, word):
        raise "Can't delete items from index."

    def __contains__(self, word):
        if not isinstance(word, str):
            raise "Presence checks only allowed with words"
        return word in self._tok_to_ind
