#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Encoder.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-07-30 10:55:18
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


class EncoderRNN(RNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
		Inputs: inputs, input_lengths
		Outputs: output, hidden
    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, num_embeddings, word2vec_size, rnn_type, rnn_size,
                 layers, dropout, brnn, embeddings=None):
        embedding_dim = word2vec_size

        super().__init__(rnn_type,
                         word2vec_size,
                         rnn_size,
                         layers,
                         True,
                         dropout,
                         brnn)
        self.hidden_size = rnn_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, Constants.PAD)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        if input_lengths:
            embedded = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if input_lengths:
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
