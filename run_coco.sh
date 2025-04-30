#!/bin/bash

file=scripts/train_coco_unia.py

nproc_per_node=$1
master_port=$2
device_pos=$3
exp_des=$4

resume_train=False
resume_path=./checkpoints/model_iter_latest.pth

CUDA_VISIBLE_DEVICES=$device_pos python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des --seed=$seed --resume_train=$resume_train --resume_path=$resume_path
