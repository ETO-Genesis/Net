#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : __init__.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-13
# Last Modified: 2019-08-13 14:58:08
# Descption    :
# Version      : Python 3.7
############################################
import argparse
from torch import nn
from Net import Constants


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 pretrained_embed=False, pretrained_weight=None):
        super().__init__()

        V = num_embeddings
        D = embedding_dim
        self.pretrained_embed = pretrained_embed
        self.pretrained_weight = pretrained_weight

        self.embedding = nn.Embedding(V, D, Constants.PAD)
        if self.pretrained_embed:
            self.embedding.weight.data.copy_(self.pretrained_weight)

    def forward(self, x):
        return self.embedding(x)


def main(args):
    """main function"""
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
