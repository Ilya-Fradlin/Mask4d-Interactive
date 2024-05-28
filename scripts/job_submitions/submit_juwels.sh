#!/bin/bash -x

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

sbatch --account=objectsegvideo --nodes=1 --ntasks-per-node=1 --ntasks-per-node=1 --cpus-per-task=4 --output=outputs/%j_test.txt --time=00-00:59:59 --partition=booster --gres=gpu:1 --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL run_on_node.sh
