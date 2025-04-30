#!/bin/bash

device=$1
inferset=$2
checkpoint=./unia/train_coco_sota_43.1/checkpoints/model_iter_80000.pth

file=./tools/coco/infer_seg_coco.py
CUDA_VISIBLE_DEVICES=$device python $file --infer_set $inferset --model_path $checkpoint