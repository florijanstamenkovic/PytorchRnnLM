# PytorchRnnLM

Language model using an RNN in PyTorch. Uses the Wiki-Text-2 long term dependency dataset:
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

## Demonstrates
1. How to use tied-embedding weights as described by: https://arxiv.org/abs/1608.05859
2. How to hack PyTorch to use PackedSequence and word embeddings without intermediary padding.

## Usage

Download the dataset and unzip file contents into the "wikitext-2" folder.
Run "main.py" with "--help" to see possible command line arguments. Default
arguments train the model on a mid-class GPU in 10 minutes to get the perplexity
score of 135 on the test-set.
