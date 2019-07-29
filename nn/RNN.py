#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : RNN.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-07-29 23:29:20
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import sys
import Constants

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    """ RNN 基础 net
    """
    def __init__(self, num_embeddings, embedding_dim, rnn_type,
                 hidden_size, layers, batch_first=True, dropout=0.1, brnn=True):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, Constants.PAD)

        self.rnn = getattr(nn, rnn_type)(
            embedding_dim,
            hidden_size,
            num_layers=layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=brnn
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
