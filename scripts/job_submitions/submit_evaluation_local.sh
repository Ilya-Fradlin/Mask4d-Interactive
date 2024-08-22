#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=en,eth,em,bond

sbatch --cpus-per-task=16 --mem=60G --partition=anchorbrew --gres=gpu:1 --time=250:00:00 --output=outputs/%j_Interactive4d_validation.txt --begin=now --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=BEGIN,END,FAIL evaluate.sh