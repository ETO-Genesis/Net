#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################################
# File Name    : manager.py
# Created By   : Suluo - sampson.suluo@gmail.com
# Creation Date: 2019-07-10
# Last Modified: 2019-08-09 23:05:44
# Descption    :
# Version      : Python 3.7
############################################
import argparse


class Manager:

    def __init__(self):
        self.registry = {}

    def register_tokenizer(self, lang: str, tokenizer):
        self.registry[lang] = tokenizer

    def tokenize(self, text: str, lang: str, **kwargs) -> list:
        tokenizer = self.registry.get(lang)
        if not tokenizer:
            raise Exception(f"No Tokenizer for language <{lang}>")
        return tokenizer.tokenize(text, **kwargs)

    def detokenize(self, tokens: list, lang: str, **kwargs) -> str:
        tokenizer = self.registry.get(lang)
        if not tokenizer:
            raise Exception(f"No Tokenizer for language <lang>")
        return tokenizer.detokenize(tokens)


def main(args):
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num', type=int, default=100, help='input num')
    args = parser.parse_args()
    main(args)
