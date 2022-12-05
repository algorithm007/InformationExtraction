#!/bin/bash
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/chinese-bert-wwm-ext
export DATA_DIR=$CURRENT_DIR/data
export OUTPUT_DIR=$CURRENT_DIR/outputs
TASK_NAME="clue"

python -m torch.distributed.launch --nproc_per_node=2 run_ner_crf.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_lower_case \
--data_dir=$DATA_DIR/${TASK_NAME}/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=24 \
--per_gpu_eval_batch_size=24 \
--learning_rate=3e-5 \
--crf_learning_rate=1e-3 \
--num_train_epochs=10 \
--logging_steps=-1 \
--save_steps=-1 \
--output_dir=$OUTPUT_DIR/${TASK_NAME}/output/ \
--overwrite_output_dir \
--seed=42
