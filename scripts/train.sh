#!/bin/bash
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO

# TRAIN
srun python main_panoptic.py