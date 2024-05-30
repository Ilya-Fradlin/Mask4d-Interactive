#!/bin/bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond,eth0
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_API_KEY=2110935b47ce429a060308845092d1bf8ce824d3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# # Install MinkowskiEngine on GPU node
# cd third_party/MinkowskiEngine 
# pip install .
# cd ../..

# run the main training script
srun python main_panoptic.py
