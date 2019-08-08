#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : cross_entropy.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-08 16:37:25
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import torch
from torch import nn
from . import Loss
from Net import Constants


class NLLLoss(Loss):
    def __init__(self, opt):
        super().__init__(opt)
        self.criterion = torch.nn.NLLLoss(ignore_index=Constants.PAD, reduction='sum')

    def forward(self, pred, gold, smoothing):
        loss = self.criterion(pred, gold)
        return loss


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
