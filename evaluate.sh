#!/usr/bin/env bash

# General Validation
python main_panoptic.py \
general.mode="validate" \
general.ckpt_path='saved/2024-02-20_183842/last.ckpt'

# # Validation with DBSCAN
# python main_panoptic.py \
# general.mode="validate" \
# general.ckpt_path='PATH_TO_CHECKPOINT.ckpt' \
# general.dbscan_eps=1.0