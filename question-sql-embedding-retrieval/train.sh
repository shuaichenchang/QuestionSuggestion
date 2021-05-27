#!/bin/bash

GPU_ID=$1
db=$2
EXPERIMENT=1
SPLIT=question-split


pretrain_model="roberta-base"
lr=2e-5
class_rate=1
batch_size=8


#for db in "imdb" # "restaurants" "scholar" "yelp" # "geography"  "advising" "atis" "academic" #
#do
    data_path=../prefix8/${SPLIT}/${db}/grouped
    table_path=../8-datasets/${db}/${db}_tables.json

    out_dir=experiment-${EXPERIMENT}/${SPLIT}/${db}/checkpoints
    mkdir -p out_dir

    model_name=model.${pretrain_model}.${lr}.${class_rate}.${batch_size}.${db}

    mkdir experiment-${EXPERIMENT}/${SPLIT}/${db}/decodes
    decode_path=experiment-${EXPERIMENT}/${SPLIT}/${db}/decodes/${model_name}.decode
    mkdir -p experiment-${EXPERIMENT}/${SPLIT}/${db}/history_questions
    history_questions=experiment-${EXPERIMENT}/${SPLIT}/${db}/history_questions/history_questions.json


    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u model.py \
    --cuda \
    --mode train \
    --pretrain_model roberta-base \
    --max_epoch 8 \
    --positive_over_negative 0.2 \
    --batch_size ${batch_size} \
    --max_input_len 400 \
    --lr ${lr} \
    --class_rate ${class_rate} \
    --table_path ${table_path} \
    --train_path ${data_path} \
    --dev_path ${data_path} \
    --test_path ${data_path} \
    --save_to ${out_dir}/${model_name} \
    --history_questions ${history_questions} \
    --decode_path ${decode_path}     > experiment-${EXPERIMENT}/${SPLIT}/${db}/${model_name}.log


  . test.sh ${GPU_ID} ${db}
#done