import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import os
import pandas as pd
from torchtext import data

def read_data(fpath):
    sents = list()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            sents.append(line.rstrip("\n"))

    return sents

def seeding(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def create_tokenizers():
    spacy_en = spacy.load('en_core_web_sm')
    
    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_vi(text):
        """
        Tokenizes Vietnamese text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    def build_vocab(src_data, trg_data, max_strlen, SRC, TRG):

        print("creating dataset and iterator... ")

        raw_data = {'src' : src_data, 'trg': trg_data}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])

        mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
        df = df.loc[mask]

        df.to_csv("translate_transformer_temp.csv", index=False)

        data_fields = [('src', SRC), ('trg', TRG)]
        dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

        SRC.build_vocab(dataset, min_freq=1)
        TRG.build_vocab(dataset, min_freq=1)

        os.remove('translate_transformer_temp.csv')


    SRC = Field(tokenize = tokenize_en,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                batch_first = True)

    TRG = Field(tokenize = tokenize_vi,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True,
                batch_first = True)
    
    train_src_data, train_trg_data = read_data("en-vi/train.en"), read_data("en-vi/train.vi")
    build_vocab(train_src_data, train_trg_data, 128, SRC, TRG)
    
    return SRC, TRG
