#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : transformer.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-13
# Last Modified: 2019-08-13 18:27:53
# Descption    :
# Version      : Python 3.7
############################################
import argparse

from . import EncoderBase

from torch import nn
from Net.nn.utils.Transformer import TransformerEmbeddings, PositionalEncoding


class EncoderTransformer(EncoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        V = self.num_embeddings
        D = self.embedding_dim = self.d_model

        self.embeddings = nn.Sequential(
            TransformerEmbeddings(V, D),
            PositionalEncoding(D, self.dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(D, self.nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def forward(self, inputs):
        embed = self.embeddings(inputs)
        out = self.transformer(embed)
        return out


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
