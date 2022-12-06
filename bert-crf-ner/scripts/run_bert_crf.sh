#!/bin/bash
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/chinese-bert-wwm-ext
export DATA_DIR=$CURRENT_DIR/data
export OUTPUT_DIR=$CURRENT_DIR/outputs
TASK_NAME="clue"

python run_bert_crf.py \
--log_dir=$CURRENT_DIR/logs \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_lower_case \
--data_dir=$DATA_DIR/${TASK_NAME}/ \
--max_seq_length=512 \
--batch_size=128 \
--learning_rate=5e-5 \
--crf_learning_rate=5e-3 \
--num_train_epochs=20 \
--logging_steps=-1 \
--save_steps=-1 \
--overwrite_output_dir \
--seed=42 \
--accelerator='gpu' \
--devices=1 \
--precision=16
