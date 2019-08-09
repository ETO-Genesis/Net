#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : tokenizers.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-10
# Last Modified: 2019-08-09 23:08:07
# Descption    :
# Version      : Python 3.7
############################################
import argparse
import re

from . import TokenizerBase
import jieba

import logging


class ZhSimple(TokenizerBase):

    """
    >>> tokenizer = ZhSimple()
    """

    SENT_SPLIT_PAT = re.compile(u'[。？！]')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _tokenize_sent(self, text, **kwargs):
        sents = []
        last_pos = 0
        for match in re.finditer(self.SENT_SPLIT_PAT, text):
            sent = text[last_pos:match.end()].strip()
            sents.append(sent)
            last_pos = match.end()
        sent = text[last_pos:].strip()
        if sent:
            sents.append(sent)
        return sents

    def _tokenize_word(self, text, **kwargs):
        res = jieba.cut(text, cut_all=False)
        return res

    def _detokenize_sent(self, sents, **kwargs):
        paragraph = "".join(sents)
        return paragraph

    def _detokenize_word(self, tokens, **kwargs):
        return "".join(tokens)


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
