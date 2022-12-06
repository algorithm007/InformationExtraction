#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project  : DjangoBlog
@File     : data_module.py
@IDE      : PyCharm
@Author   : 算法小学僧
@Date     : 2022/12/5 23:18 
'''
import pytorch_lightning as pl
from transformers import BertTokenizer
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader, DistributedSampler

from preprocess.data_preprocess import load_and_cache_examples
from tools.helpers import collate_fn


class DataModule(pl.LightningDataModule):
    def __init__(self, ner_logger, **kwargs):
        super(DataModule, self).__init__()

        self.kwargs = kwargs

        model_name_or_path = kwargs.get("model_name_or_path", None)
        do_lower_case = kwargs.get("do_lower_case", None)
        self.task_name = kwargs.get("task_name", None)

        self.batch_size = self.kwargs.get('batch_size')

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.logger = ner_logger

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = load_and_cache_examples(self.kwargs, self.task_name, self.tokenizer,
                                                    data_type="train", logger=self.logger)
            self.dev_dataset = load_and_cache_examples(self.kwargs, self.task_name, self.tokenizer,
                                                  data_type="dev", logger=self.logger)
        else:
            self.test_dataset = load_and_cache_examples(self.kwargs, self.task_name, self.tokenizer,
                                                   data_type="test", logger=self.logger)

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        dev_sampler = SequentialSampler(self.dev_dataset)
        return DataLoader(self.dev_dataset, sampler=dev_sampler, batch_size=self.batch_size,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_dataset)
        return DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.batch_size,
                          collate_fn=collate_fn)


















































