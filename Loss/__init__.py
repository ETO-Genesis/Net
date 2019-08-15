#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : __init__.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-29
# Last Modified: 2019-08-11 19:56:46
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import torch
from torch import nn
from Net import Constants


def cal_correct(pred, gold):
    """cal correct"""
    pred = torch.argmax(pred, dim=-1)
    non_pak_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pak_mask).sum().item()
    return n_correct


class Loss(nn.Module):
    """Loss 基类"""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def cal_performance(self, pred, gold):
        """cal loss and correct"""
        loss = self.forward(pred, gold)
        n_correct = cal_correct(pred, gold)
        return loss, n_correct

    def forward(self, pred, gold):
        """cal Loss"""
        raise NotImplementedError()


def main(args):
    """main"""
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
