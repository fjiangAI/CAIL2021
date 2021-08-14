#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5p                 #
#                                                                   #
#                   @File Name    : tokenizer.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2021/3/24 16:31                 #
#                                                                   #
#                   @Last Update  : 2021/3/24 16:31                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''

import jieba
from transformers import BertTokenizer


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens