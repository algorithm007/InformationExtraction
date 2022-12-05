#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project  : DjangoBlog
@File     : data_processor.py
@IDE      : PyCharm
@Author   : 算法小学僧
@Date     : 2022/12/5 09:22 
'''
import os
import json
import copy

import torch
import torch.distributed
from tqdm import tqdm
from torch.utils.data import TensorDataset


class InputExample(object):
    def __init__(self, guid, text_a, labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class data converters for sequence classification data sets"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(self, input_file):
        total_data = []
        with open(input_file, 'r', encoding="utf-8") as f:
            for line in tqdm(f, desc=f"reading {input_file}"):
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_idx in value.items():
                            for start_idx, end_idx in sub_idx:
                                assert text[start_idx:end_idx + 1] == sub_name
                                if start_idx == end_idx:
                                    labels[start_idx] = 'S-' + key
                                else:
                                    labels[start_idx] = 'B-' + key
                                    labels[start_idx + 1:end_idx + 1] = ['I-' + key] * (end_idx - start_idx)
                total_data.append({"words": words, "labels": labels})
        return total_data


class ClueProcessor(DataProcessor):
    """Processor for clue dataset"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'dev.json')), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, 'test.json')), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets"""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f'{set_type}-{i}'
            text_a = line["words"]
            labels = line["labels"]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    def get_labels(self, data_dir):
        if not os.path.exists(os.path.join(data_dir, "labels.json")):
            label_list = set(['X', '[START]', '[END]'])
            examples = self.get_train_examples(data_dir)
            for example in examples:
                label = example.labels
                for l in label:
                    label_list.add(l)
            with open(os.path.join(data_dir, "labels.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": list(label_list)}, f, indent=2, ensure_ascii=False)
        else:
            label_list = json.load(open(os.path.join(data_dir, "labels.json"), "r", encoding="utf-8"))["labels"]
        return list(label_list)


def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length, logger=None):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)

        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # [CLS] [SEP]
        special_tokens_count = 2
        tokens = tokens[:max_seq_length-special_tokens_count]
        label_ids = label_ids[:max_seq_length-special_tokens_count]


        # The convertion in BERT is:
        # (a) For sequence pairs:
        # tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        # type_ids: 0    0  0    0    0       0    0  0    1  1  1   1  1  1
        # (b) For single sequence:
        # tokens:  [CLS] the dog is hairy . [SEP]
        # type_ids:  0   0    0  0   0    0   0
        sep_token = "[SEP]"
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [0] * len(tokens)

        cls_token = "[CLS]"
        tokens = [cls_token] + tokens
        label_ids = [label_map['O']] + label_ids
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        input_len = len(label_ids)

        # padding
        pad_token_id = 0
        padding_length = max_seq_length - input_len
        input_ids += [pad_token_id] * padding_length
        input_mask += [pad_token_id] * padding_length
        segment_ids += [pad_token_id] * padding_length
        label_ids += [pad_token_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      input_len=input_len,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
    return features


def load_and_cache_examples(args, task_name, tokenizer, data_type="train", logger=None):
    if args.local_rank not in [-1, 0] and data_type != "evaluate":
        torch.distributed.barrier()
    processor = ner_processors[task_name]()
    # load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_crf_{}_{}_{}_{}".format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task_name)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        label_list = processor.get_labels(args.data_dir)

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == "train" else \
                                                    args.eval_max_seq_length,
                                                logger=logger)
        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and data_type != "evaluate":
        torch.distributed.barrier()

    # convert to tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features]).long()
    all_input_mask = torch.tensor([f.input_mask for f in features]).long()
    all_segment_ids = torch.tensor([f.segment_ids for f in features]).long()
    all_label_ids = torch.tensor([f.label_ids for f in features]).long()
    all_lens = torch.tensor([f.input_len for f in features]).long()

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_lens)
    return dataset


ner_processors = {
    "clue": ClueProcessor,
}
