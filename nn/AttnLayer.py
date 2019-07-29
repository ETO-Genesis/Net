#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : AttnLayer.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-03-07
# Last Modified: 2019-07-29 23:00:38
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import time
import os
import sys
import torch
import torch.nn.functional as F


class GlobalAttention(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = torch.nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, ht, enc_outputs):
        """
        Args:
            ht: (1, batch, 512)
            enc_outputs: (batch, seq_len, 512): 所有的hs
        Return:
            context: (batch, 512)
            attn_weights: (batch, 1, seq_len)
            # 输出是注意力的概率，也就是长度为input_lengths的向量，它的和加起来是1。
        """
        attn_energies = self.score(ht.transpose(0, 1), enc_outputs)

        # (batch, seq_len) -> (batch, 1, seq_len)
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        context = attn_weights.bmm(enc_outputs)
        return context, attn_weights

    def score(self, ht, enc_outputs):
        if self.method == 'dot':
            energy = torch.sum(ht * enc_outputs, dim=2)
        elif self.method == 'general':
            energy = self.attn(enc_outputs)
            energy = torch.sum(ht * energy, dim=2)
        elif self.method == 'concat':
            energy = self.attn(torch.cat((ht.expand(enc_outputs.size(0), -1, -1), enc_outputs), 2)).tanh()
            energy = torch.sum(self.other * energy, dim=2)
        return energy


def main(args):
    return args.num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
