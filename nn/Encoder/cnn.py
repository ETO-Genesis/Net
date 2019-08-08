#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : cnn.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-08 19:05:11
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import sys

from . import EncoderBase
from Net import Constants

from torch import nn


class EncoderCNN(EncoderBase):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Kwargs:
    Args:
		Inputs: inputs, lengths
		Outputs: output, hidden
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        V = self.num_embeddings
        D = self.embedding_dim
        Ci = 1
        Co = self.kernel_num
        Ks = self.kernel_sizes

        self.hidden_size = self.rnn_size
        self.embedding = nn.Embedding(V, D, Constants.PAD)
        if self.pretrained_embed:
            self.embedding.weight.data.copy_(self.pretrained_weight)

        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(Ci, Co, (K, D))
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)]
            for K in Ks
        )


    def forward(self, inputs):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input (batch, seq_len):
        Returns: output, hidden
            output:
            hidden:
        """
        embedded = self.embedding(inputs)
        x = embedded.view(x.size(0), 1, self.max_len, self.embedding_dim)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return outputs


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
