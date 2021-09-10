#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5p                 #
#                                                                   #
#                   @File Name    : data_set_joint.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2021/3/27 21:19                 #
#                                                                   #
#                   @Last Update  : 2021/3/27 21:19                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''

import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class GPT2NewsTitleDataSet(Dataset):
    """新闻标题生成模型所需要的数据类"""

    def __init__(self, tokenizer, max_len, rs_max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        初始化函数
        Args:
            tokenizer: 分词器
            max_len: 数据的最大长度
            rs_max_len: 生成标题的最大长度
            data_dir: 保存缓存文件的路径
            data_set_name: 数据集名字
            path_file: 原始数据文件
            is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer = tokenizer
        self.du1_id = self.tokenizer.convert_tokens_to_ids("[du1]")
        self.rs_id = self.tokenizer.convert_tokens_to_ids("[rs]")
        self.du2_id = self.tokenizer.convert_tokens_to_ids("[du2]")
        self.max_len = max_len
        self.rs_max_len = rs_max_len
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        """
        加载原始数据，生成数据处理后的数据
        Args:
            path_file: 原始数据路径

        Returns:

        """
        self.data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
                # 使用convert_feature函数，对新闻正文和标题进行索引化，生成模型所需数据格式
                input_ids, label_ids, class_label = self.convert_feature(sample)
                self.data_set.append({"input_ids": input_ids, "label_ids": label_ids, "class_ids": class_label})
        return self.data_set

    def find_longest_du(self, dus):
        max_length = 0
        max_index = -1
        all_length = 0
        for index, du in enumerate(dus):
            all_length += len(du)
            if len(du) > max_length:
                max_length = len(du)
                max_index = index
        return max_index, all_length

    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 一个字典，包含新闻的正文和新闻的标题，格式为{"content": content, "title": title}

        Returns:

        """
        input_ids = []
        label_ids = []
        # 对新闻正文进行tokenizer.tokenize分词
        du1_tokens = self.tokenizer.tokenize(sample["du1"])
        rs_tokens = self.tokenizer.tokenize(sample["rs"])
        du2_tokens = []
        for du2 in sample["du2"]:
            du2_token = self.tokenizer.tokenize(du2)
            du2_tokens.append(du2_token)
        # 判断如果标题过长，进行截断
        if len(rs_tokens) > self.rs_max_len:
            rs_tokens = rs_tokens[:self.rs_max_len]
        # 判断如果正文过长，进行截断
        max_index, all_length = self.find_longest_du(du2_tokens)
        while len(du1_tokens) + all_length > self.max_len - len(rs_tokens) - (3 + len(du2_tokens)):
            if len(du1_tokens) > len(du2_tokens[max_index]):
                du1_tokens = du1_tokens[:-1]
            else:
                du2_tokens[max_index] = du2_tokens[max_index][:-1]
            max_index, all_length = self.find_longest_du(du2_tokens)
        # 生成模型所需的input_ids和token_type_ids
        input_ids.append(self.tokenizer.cls_token_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(du1_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        for du2_token in du2_tokens:
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(du2_token))
            input_ids.append(self.tokenizer.sep_token_id)
        label_ids.extend(self.tokenizer.convert_tokens_to_ids(rs_tokens))
        label_ids.append(self.tokenizer.sep_token_id)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids + label_ids) <= self.max_len
        class_ids = sample["label"]
        return input_ids, label_ids, class_ids

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据

    Returns:

    """
    batch_size = len(batch_data)
    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}
    input_ids_list, label_ids_list, class_list = [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        label_ids_temp = instance["label_ids"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        label_ids_list.append(torch.tensor(label_ids_temp, dtype=torch.long))
        class_list.append(instance["class_ids"])
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "label_ids": pad_sequence(label_ids_list, batch_first=True, padding_value=0),
            "class_ids": torch.tensor([f for f in class_list], dtype=torch.long)}
