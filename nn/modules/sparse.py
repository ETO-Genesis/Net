#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : sparse.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-16
# Last Modified: 2019-08-16 15:55:52
# Descption    :
# Version      : Python 3.7
############################################
import argparse
from torch import nn
from Net import Constants


class Embedding(nn.Module):
    def __init__(self, V, D, pretrained_embed=False, pretrained_weight=None):
        super().__init__()
        self.embedding = nn.Embedding(V, D, Constants.PAD)
        if pretrained_embed:
            self.embedding.weight.data.copy_(pretrained_weight)

    def forward(self, x):
        return self.embedding(x)


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
