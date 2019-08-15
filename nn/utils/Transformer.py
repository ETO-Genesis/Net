#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Transformer.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-13
# Last Modified: 2019-08-13 19:32:37
# Descption    :
# Version      : Python 3.7
############################################
import argparse

import math
import torch
from torch import nn
from . import Embeddings


class TransformerEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim,
                 pretrained_embed=False, pretrained_weight=None):
        super().__init__(num_embeddings, embedding_dim,
                         pretrained_embed, pretrained_weight)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
