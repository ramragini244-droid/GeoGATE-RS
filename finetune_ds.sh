#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6012

MODEL="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V"

# # git10M
# DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/Git10M/git10m_output.json"

# # rscid
# # DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/Git10M/git10m_output.json"
# EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/RSICD_data/update_json/rsicd转换后.json"

# # FAIR1M 和 NWPU_VHR-10 数据集
# DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/Code/target_detection/down_load_data/FAIR1M/fair1m_conversations.json" # 训练集路径
# EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/Code/target_detection/down_load_data/NWPU_VHR-10_dataset/train_set.json" # 测试集路径

# # VRSBench 数据集
# DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/VRSBench/vrsbench_train_set.json" # 训练集路径
# EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/VRSBench/vrsbench_val_set.json" # 测试集路径

# # xBD 数据集
# DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/xBD_Xview/xBD_train_converted.json" # 训练集路径
# EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/xBD_Xview/xBD_test_converted.json" # 测试集路径

# # total 数据集
# DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/MME-RealWorld/MME_RealWord_converted_data.json" # 训练集路径
# EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/MME-RealWorld/MME_RealWord_converted_data.json" # 测试集路径

# total 数据集
DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/merged_train_id_unique.json" # 训练集路径
EVAL_DATA="/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/merged_test_id_unique.json" # 测试集路径




MODEL_MAX_Length=4096 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --output_dir output/total \
    --logging_dir output/total \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" 
