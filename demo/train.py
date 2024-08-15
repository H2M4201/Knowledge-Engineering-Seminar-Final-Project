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
import model

import random
import math
import time

from model import Encoder, Decoder, Seq2Seq

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

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

def read_data(fpath):
    sents = list()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            sents.append(line.rstrip("\n"))

    return sents

train_src_data, train_trg_data = read_data("en-vi/train.en"), read_data("en-vi/train.vi")
valid_src_data, valid_trg_data = read_data("en-vi/valid.en"), read_data("en-vi/valid.vi")
test_src_data, test_trg_data = read_data("en-vi/test.en"), read_data("en-vi/test.vi")

import os

import pandas as pd
from torchtext import data


def create_dataset(src_data, trg_data, max_strlen, SRC, TRG, is_train=True):

    print("creating dataset and iterator... ")

    raw_data = {'src' : src_data, 'trg': trg_data}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    dataset = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    if is_train:
        SRC.build_vocab(dataset, min_freq=1)
        TRG.build_vocab(dataset, min_freq=1)

    os.remove('translate_transformer_temp.csv')

    return dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Finally, we define the device and the data iterator."""

train_dataset = create_dataset(train_src_data, train_trg_data, 128, SRC, TRG, is_train=True)
valid_dataset = create_dataset(valid_src_data, valid_trg_data, 128, SRC, TRG, is_train=False)
test_dataset = create_dataset(test_src_data, test_trg_data, 128, SRC, TRG, is_train=False)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
                                                  (train_dataset, valid_dataset, test_dataset),
                                                  batch_size=32,
                                                  device=device,
                                                  sort_key=lambda x: (len(x.src), len(x.trg))
                                                )

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              max_length=256)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              max_length=256)

"""Then, use them to define our whole sequence-to-sequence encapsulating model."""

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:,:-1])

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

