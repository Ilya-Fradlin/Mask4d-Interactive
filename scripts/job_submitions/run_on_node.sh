#!/bin/bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond,eth0
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_API_KEY=2110935b47ce429a060308845092d1bf8ce824d3
export CUDA_VISIBLE_DEVICES=0,1,2,3


# debugging options:
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export GLOO_LOG_LEVEL=DEBUG
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# # Install MinkowskiEngine on GPU node
# cd third_party/MinkowskiEngine 
# pip install .
# cd ../..

# run the main training script
# srun python main_panoptic.py
srun python main_panoptic.py data.voxel_size=0.8 data.batch_size=2 data.num_workers=1 general.experiment_name=debugging-with-logging