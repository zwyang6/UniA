#!/bin/bash

file=./scripts/train_voc_unia.py

nproc_per_node=$1
master_port=$2
device_pos=$3
exp_des=$4

CUDA_VISIBLE_DEVICES=$device_pos python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des