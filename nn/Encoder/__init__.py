#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : __init__.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-08-08
# Last Modified: 2019-08-09 00:28:40
# Descption    :
# Version      : Python 3.7
############################################
import argparse

from torch import nn


class EncoderBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def forward(self, src, lengths=None):
        raise NotImplementedError


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
