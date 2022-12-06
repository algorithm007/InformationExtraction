#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project  : DjangoBlog
@File     : model_module.py
@IDE      : PyCharm
@Author   : 算法小学僧
@Date     : 2022/12/6 10:04 
'''
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
from transformers import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from models.bert_for_ner import BertCrfForNer
from preprocess.data_preprocess import ner_processors as processor
# from tools.ner_metrics import SeqEntityScore
from seqeval.metrics import f1_score, recall_score, precision_score


class BertCrfNer(pl.LightningModule):
    def __init__(self, ner_logger, **kwargs):
        super(BertCrfNer, self).__init__()

        model_name_or_path = kwargs.get("model_name_or_path", None)
        task_name = kwargs.get("task_name", None)
        data_dir = kwargs.get("data_dir", None)
        markup = kwargs.get("markup", None)

        self.weight_decay = kwargs.get("weight_decay", None)
        self.learning_rate = kwargs.get("learning_rate", None)
        self.crf_learning_rate = kwargs.get("crf_learning_rate", None)
        self.num_train_epochs = kwargs.get("num_train_epochs", None)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", None)
        self.adam_epsilon = kwargs.get("adam_epsilon", None)
        self.warmup_proportion = kwargs.get("warmup_proportion", None)

        label_list = processor[task_name]().get_labels(data_dir)
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

        config = BertConfig.from_pretrained(model_name_or_path, num_labels=len(label_list))
        self.model = BertCrfForNer.from_pretrained(model_name_or_path, config=config)
        # self.metrics = SeqEntityScore(self.id2label, markup=markup)
        self.model_logger = ner_logger

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, label_lens = batch
        input = {"input_ids": input_ids, "attention_mask": input_mask,
                 "token_type_ids": segment_ids, "labels": label_ids}
        outputs = self(**input)
        loss = outputs[0]
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        mean_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_dict({"train_loss": mean_train_loss})

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids, label_lens = batch
        input = {
            "input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids, "labels": label_ids
        }
        outputs = self(**input)
        val_loss, logits = outputs[:2]
        tags = self.model.crf.decode(logits, input["attention_mask"])
        out_label_ids = input["labels"].cpu().numpy().tolist()
        input_lens = label_lens.cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()

        batch_true, batch_pred = [], []
        for i, label in enumerate(out_label_ids):
            y_true, y_pred = [], []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    batch_true.append(y_true)
                    batch_pred.append(y_pred)
                    break
                else:
                    y_true.append(self.id2label[out_label_ids[i][j]])
                    y_pred.append(self.id2label[tags[i][j]])

        f1 = f1_score(batch_true, batch_pred)
        recall = recall_score(batch_true, batch_pred)
        pre = precision_score(batch_true, batch_pred)

        f1 = torch.tensor([f1]).float()
        recall = torch.tensor([recall]).float()
        pre = torch.tensor(pre).float()
        return {"precision": pre, "recall": recall, "f1": f1, "val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_f1 = torch.stack([x["f1"] for x in outputs]).mean()
        val_precision = torch.stack([x["precision"] for x in outputs]).mean()
        val_recall = torch.stack([x["recall"] for x in outputs]).mean()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.model_logger.info({"val_f1": val_f1.item(), "val_precision": val_precision.item(),
                                "val_recall": val_recall.item(), "val_loss": val_loss.item()})
        self.log_dict({"val_f1": val_f1, "val_precision": val_precision, "val_recall": val_recall, "val_loss": val_loss})

    def setup(self, stage=None):
        if stage != 'fit':
            return
        train_dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
        self.total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        crf_param_optimizer = list(self.model.crf.named_parameters())
        linear_param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            # bert
            {
                "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {
                "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                "lr": self.learning_rate,
            },

            # crf
            {
                "params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.crf_learning_rate,
            },
            {
                "params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.crf_learning_rate,
            },

            # linear
            {
                "params": [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.crf_learning_rate,
            },
            {
                "params": [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.crf_learning_rate,
            }]
        warmup_steps = int(self.total_steps * self.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=self.total_steps)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        early_stopping = plc.EarlyStopping(monitor='val_f1', mode='max', patience=6, min_delta=0.001)
        model_checkpoint = plc.ModelCheckpoint(monitor="val_f1", filename="bert-{epoch:02d}-{val_f1:.3f}",
                                               save_top_k=1, mode='max', save_last=False)
        return [early_stopping, model_checkpoint]





















































