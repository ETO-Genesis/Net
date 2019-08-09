#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : __init__.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-10
# Last Modified: 2019-08-09 23:08:26
# Descption    :
# Version      : Python 3.7
############################################
import argparse


class TokenizerBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _tokenize_sent(self, text, **kwargs):
        raise NotImplementedError()

    def _tokenize_word(self, text, **kwargs):
        raise NotImplementedError()

    def _detokenize_sent(self, sents, **kwargs):
        raise NotImplementedError()

    def _detokenize_word(self, tokens, **kwargs):
        raise NotImplementedError()

    def tokenize(self, text: str) -> list:
        tokens = []
        for sent in self._tokenize_sent(text):
            tokens.extend(self._tokenize_word(sent))
        return tokens

    def detokenize(self, sentences_tokens):
        """Return the best one in string format
        """
        sentences = []
        for sentences_nbest_tokens in sentences_tokens:
            best_tokens = sentences_nbest_tokens[0]
            best_sent = self._detokenize_word(best_tokens)
            sentences.append(best_sent)
        return sentences


from .tokenizers import ZhSimple
from .manager import Manager

tokenizer_manager = Manager()
tokenizer_manager.register_tokenizer('zh', ZhSimple())
