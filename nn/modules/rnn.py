#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : rnn.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-16
# Last Modified: 2019-08-17 11:56:18
# Descption    :
# Version      : Python 3.7
############################################
import argparse
from torch import nn


class RNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers,
                 batch_first=True, dropout=0.1, bidirectional=False):
        super().__init__()

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, embed, hidden=None):
        output, hidden = self.rnn(embed, hidden)
        return output, hidden


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='inputnum')
    args = parser.parse_args()
    main(args)
