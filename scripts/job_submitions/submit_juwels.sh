#!/bin/bash -x

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# srun --account=objectsegvideo --ntasks-per-node=1 --nodes=1 --gres=gpu:1 --time=00:20:00 --partition=booster --pty bash -l
# sbatch --account=objectsegvideo --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --output=outputs/%j_preprocessing.txt --time=00-02:59:59 --partition=booster --gres=gpu:0 --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL scripts/job_submitions/run_preprocessing.sh
sbatch --account=objectsegvideo --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --output=outputs/%j_train_things_only_test.txt --time=00-10:59:59 --partition=booster --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL scripts/job_submitions/run_on_node.sh
