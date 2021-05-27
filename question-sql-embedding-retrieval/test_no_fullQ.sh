#!/bin/bash

GPU_ID=$1
data_path="split_labeled"



pretrain_model="roberta-base"
lr=2e-5
class_rate=1
batch_size=40
model_name=model.nofullQ.${pretrain_model}.${lr}.${class_rate}.${batch_size}
decode_path="decodes/"${model_name}".decode"
save_to="checkpoints/"${model_name}"/"
history_questions=${data_path}"/history_questions.json"
best_epoch=0

CUDA_VISIBLE_DEVICES=$GPU_ID python -u model.py \
--cuda \
--mode test \
--pretrain_model roberta-base \
--max_epoch 5 \
--lr ${lr} \
--class_rate ${class_rate} \
--batch_size ${batch_size} \
--max_input_len 300 \
--train_path ${data_path} \
--dev_path ${data_path} \
--test_path ${data_path} \
--load_model_path ${save_to}epoch_${best_epoch}.pkl \
--history_questions ${history_questions} \
--decode_path ${decode_path} 1>>logs/${model_name}.log

