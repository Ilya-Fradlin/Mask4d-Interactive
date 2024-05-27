#!/usr/bin/env bash

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

export NCCL_SOCKET_IFNAME=en,eth,em,bond
export CUDA_LAUNCH_BLOCKING=1

sbatch --partition=a40-lo -c 16 --gres=gpu:1 --mem=48G --job-name=interactive4d --time=10-00:00:00 \
--signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL --output=outputs/%j_5clickslimit10_full_click_1gpu.txt scripts/train.sh

sbatch --partition=a40-lo -c 32 --gres=gpu:2 --ntasks-per-node=2 --mem=96G --job-name=multi_training --time=10-00:00:00 \
--signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL --output=outputs/%j_5clicks_full_2gpu.txt scripts/train.sh
