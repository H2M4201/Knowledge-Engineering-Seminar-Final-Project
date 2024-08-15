import streamlit as st
import torch
import matplotlib.pyplot as plt
from pyngrok import ngrok
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field
from torchtext.data import BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import utils
from model import Encoder, Decoder, Seq2Seq

# Load pre-trained model and tokenizer
@st.cache_resource 
def load_model():
    utils.seeding(SEED=1234)
    
    device = torch.device('cpu')

    SRC, TRG = utils.create_tokenizers()
    
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


    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    
    model.load_state_dict(torch.load('tut6-model-2.pt', map_location=torch.device('cpu')))
    
    return model, SRC, TRG, device

@st.cache_data
def load_tokenizer():
  return spacy.load('en_core_web_sm')

# Streamlit app
def main():
    st.title("English to Vietnamese Translation with Attention Visualization")
    text = st.text_area("Enter English text:", value="Hello, how are you?")
    nlp = load_tokenizer()
    text = [token.text.lower() for token in nlp(text)]

    if st.button("Translate"):
        translation, attention = translate_sentence(text)
        st.success(f"Translation: {' '.join(translation[:-1])}")
        st.subheader("Attention map visualization of each head")
        display_attention(text, translation, attention)

# Translation function
def translate_sentence(text, max_len = 128):
    tokens = text
    model, src_field, trg_field, device = load_model()
    
    model.eval()

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15,25))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                           rotation=45)
        ax.set_yticklabels(['']+translation)
        ax.title.set_text(f"Attention map of head {i}")
        ax.title.set_position([0.5, -0.15])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    st.pyplot(fig)

if __name__ == "__main__":
    main()
