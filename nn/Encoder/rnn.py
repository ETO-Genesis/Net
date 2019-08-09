#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : rnn.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-08 22:55:02
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class EncoderRNN(EncoderBase):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Kwargs:
        num_embeddings
        word2vec_size
        rnn_type
        rnn_size
        layers
        dropout
        brnn

        pretrained_embed
    Args:
        Inputs: inputs, input_lengths
        Outputs: output, hidden
    Examples:
         >>> encoder = EncoderRNN(**kwargs)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        V = self.num_embeddings = self.vocab_size
        D = self.embedding_dim

        self.hidden_size = self.rnn_size
        self.embedding = nn.Embedding(V, D, Constants.PAD)
        if self.pretrained_embed:
            self.embedding.weight.data.copy_(self.pretrained_weight)

        self.rnn = getattr(nn, self.rnn_type)(
            input_size=D,
            hidden_size=self.rnn_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.brnn
        )

    def forward(self, inputs, lengths=None, hidden=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input (batch, seq_len):
            lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            output:
            hidden:
        """
        embedded = self.embedding(inputs)
        if lengths:
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, hidden = self.rnn(embedded, hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        outputs = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return outputs, hidden


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
