#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : crf.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-08 21:58:30
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


class CRFLoss(Loss):
    def __init__(self, opt):
        super().__init__(opt)
        num_tags = opt.tgt_vocab_size
        self.crf = CRF(num_tags, batch_first=True).to(opt.device)

    def forward(self, pred, gold, smotthing):
        loss = -self.crf(pred, gold, mask=Constants.PAD)
        return loss


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
