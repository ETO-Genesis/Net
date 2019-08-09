#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : cross_entropy.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-09 13:20:15
# Descption    :
# Version      : Python 3.7
############################################
import argparse
from torch import nn
from . import Loss
from Net import Constants


class NLLLoss(Loss):
    def __init__(self, opt):
        super().__init__(opt)
        self.criterion = nn.NLLLoss(ignore_index=Constants.PAD, reduction='sum')

    def forward(self, pred, gold, smoothing):

        pred = pred.contiguous().view(-1, pred.size(2))
        gold = gold.contiguous().view(-1)
        loss = self.criterion(pred, gold)
        return loss

    def cal_performance(self, pred, gold, smoothing):
        gold = gold[:, 1]
        loss, n_correct = super().cal_performance(gold, gold, smoothing)
        return loss, n_correct


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
