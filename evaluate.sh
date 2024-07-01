#!/usr/bin/env bash

# General Validation
srun python main_panoptic.py

# # Validation with DBSCAN
# python main_panoptic.py \
# general.mode="validate" \
# general.ckpt_path='PATH_TO_CHECKPOINT.ckpt' \
# general.dbscan_eps=1.0