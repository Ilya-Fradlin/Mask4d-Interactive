#!/usr/bin/env bash

export NCCL_SOCKET_IFNAME=en,eth,em,bond
export CUDA_LAUNCH_BLOCKING=1

# if [ "$#" -ne 11 ]; then
#     echo "Usage ${0} <port_id> <data_file> <ann_sup> <training_sup> <path_pretrain_model> <stage> <num_workers> <lr> <batch_size> <iterations> <steps>"
#     exit
# fi

# PORT_ID=$1
# DATA_FILE=$2
# ANN_SUP=$3
# TRAIN_SUP=$4
# PATH_PRETRAIN_MODEL=$5
# STAGE=$6
# NUM_WORKERS=$7
# LR=$8
# BATCH_SIZE=$9
# NUM_ITERATIONS=${10}
# STEPS=${11}

# sbatch --partition=trtx-lo -c 4 --gres=gpu:1 --mem=24G --job-name=train_agile3d --time=6-00:00:00 \
# --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL --output=outputs/output_%j.txt train_multi_scannet40.sh

sbatch --partition=a40-lo -c 16 --gres=gpu:1 --mem=48G --job-name=interactive4d_training --time=8-00:00:00 \
--signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL --output=outputs/%j_training_full_1gpu.txt scripts/train.sh
