#!/bin/bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond
export CUDA_LAUNCH_BLOCKING=1

# run the main training script
python datasets/preprocessing/semantic_kitti_obj_preprocessing.py preprocess
