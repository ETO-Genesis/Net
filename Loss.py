#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : Loss.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-07-29 23:37:44
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import torch
from torchcrf import CRF
import torch.nn.functional as F
import Constants


class Loss(object):
    def __init__(self, opt):
        pass

    def cal_performance(self, pred, gold, smoothing=False):
        loss = self.cal_loss(pred, gold, smoothing)
        n_correct = self.cal_correct(pred, gold)

    def cal_loss(self, pred, gold, smoothing=False):
        raise NotImplementedError()

    def cal_correct(self, pred, gold):
        pred = torch.argmax(pred, dim=-1)
        non_pak_mask = gold.ne(Constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pak_mask).sum().item()
        return n_correct


class CRFLoss(Loss):
    def __init__(self, opt, num_tags):
        super().__init__(opt)
        self.crf = CRF(num_tags, batch_first=True).to(opt.device)

    def cal_loss(self, pred, gold, smotthing):
        loss = -self.crf(pred, gold, mask=Constants.PAD)
        return loss


class CrossEntropy(Loss):
    def __init__(self, opt):
        super().__init__(opt)

    def cal_loss(self, pred, gold, smoothing):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(Constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

        return loss


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
