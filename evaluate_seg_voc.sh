#!/bin/bash

device=0
inferset=val
file=tools/voc/infer_seg.py

checkpoint=/data/PROJECTS/UniA_2024/TMM_2024/代码公示/最终代码上传/UniA/pretrain/voc/checkpoints/model_iter_20000.pth
CUDA_VISIBLE_DEVICES=$device python $file --infer_set $inferset --model_path $checkpoint
