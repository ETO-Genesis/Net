#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : rnn.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-13
# Last Modified: 2019-08-09 10:10:02
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import DecoderBase

from Net.nn.Module.AttnLayer import GlobalAttention
from Net import Constants


class DecoderRNN(DecoderBase):
    r"""
    Provides functionality for decoding in a seq2seq framework, \
        with an option for attention.
    Args:
    Attributes:
    Inputs: inputs, encoder_hidden, encoder_outputs
    Outputs:
        dec_ouputs: 概率矩阵
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        V = self.num_embeddings = self.vocab_size
        D = self.embedding_dim

        self.embedding = nn.Embedding(V, D, Constants.PAD)
        print(kwargs)
        if self.pretrained_embed:
            self.embedding.weight.data.copy_(self.pretrained_weight)

        hidden_size = self.rnn_size

        self.rnn = getattr(nn, self.rnn_type)(
            input_size=D,
            hidden_size=self.rnn_size,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.brnn
        )

        self.attn = GlobalAttention(self.method, hidden_size)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, V)
        )
        self.teacher_forcing_ratio = 0

    def forward(self, inputs=None, enc_hidden=None, enc_outputs=None):
        device = self.device

        batch_size = self._validate_args(inputs, enc_hidden)

        outputs = torch.empty(batch_size, 1, self.vocab_size, device=device)

        last_hidden = enc_hidden
        last_context = torch.zeros(batch_size, self.rnn_size, device=device)
        dec_input = inputs[:, 0].unsqueeze(1)

        use_teacher_forcing = random.random() < self.teacher_forcing_ratio

        max_lenth = self.max_lenth if inputs is None else inputs.size(1)
        for t in range(1, max_lenth - 1):
            output, last_context, last_hidden = self.step(
                dec_input, last_context, last_hidden, enc_outputs
            )
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)

            if use_teacher_forcing:
                dec_input = inputs[:, t]
            else:
                _, dec_input = output.topk(1)
                # !teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值
        return outputs

    def step(self, word, last_context, last_hidden, enc_outputs):
        word_embed = self.embedding(word)

        rnn_output, hidden = self.rnn(word_embed, last_hidden)
        context, attn = self.attn(hidden[:1], enc_outputs)

        rnn_output = rnn_output.squeeze()
        context = context.squeeze()

        concat_input = torch.cat((rnn_output, context), 1)
        # out是(512, 词典大小=vocab_size)
        output = self.out(concat_input)
        # 用softmax变成概率，表示当前时刻输出每个词的概率。(B, output_size)
        output = F.log_softmax(output, dim=1)
        return output, context, hidden

    def _validate_args(self, inputs, enc_hidden):
        if inputs is not None:
            batch_size = inputs.size(0)
        else:
            if self.rnn_type == "LSTM":
                batch_size = enc_hidden[0].size(1)
            elif self.rnn_type == "GRU":
                batch_size = enc_hidden.size(1)
        return batch_size


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
