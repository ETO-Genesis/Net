#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : __init__.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-08 16:39:31
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import torch
from torch import nn
from Net import Constants


class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()

    def cal_performance(self, pred, gold, smoothing=False):
        loss = self.forward(pred, gold, smoothing)
        n_correct = self.cal_correct(pred, gold)
        return loss, n_correct

    def forward(self, pred, gold, smoothing=False):
        raise NotImplementedError()

    def cal_correct(self, pred, gold):
        pred = torch.argmax(pred, dim=-1)
        non_pak_mask = gold.ne(Constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pak_mask).sum().item()
        return n_correct


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
