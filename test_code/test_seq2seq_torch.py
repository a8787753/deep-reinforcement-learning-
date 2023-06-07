import torch
import torch.nn
import torch.optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

# print(torch.__version__)
# print(torch.cuda.is_available())

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    # Tokenizes German text from a string into a list of strings (tokens) and reverses it
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    # Tokenizes English text from a string into a list of strings (tokens)
    return [tok.text for tok in spacy_en.tokenizer(text)][::-1]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

