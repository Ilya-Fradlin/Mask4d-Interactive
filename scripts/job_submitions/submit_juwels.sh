#!/bin/bash -x

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


module load Stages/2023
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.7
module load Python/3.10.4
module load PyTorch/1.12.0-CUDA-11.7
module load PyTorch-Lightning
pyenv activate interactive4d

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# srun --account=objectsegvideo --ntasks-per-node=1 --nodes=1 --gres=gpu:1 --time=00:20:00 --partition=booster --pty bash -l
# sbatch --account=objectsegvideo --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --output=outputs/%j_preprocessing.txt --time=00-02:59:59 --partition=booster --gres=gpu:0 --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=FAIL scripts/job_submitions/run_preprocessing.sh

sbatch --account=objectsegvideo --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --output=outputs/%j_baseline.txt --time=00-23:59:59 --partition=booster --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=--mail-type=BEGIN,END,FAIL scripts/job_submitions/run_on_node.sh
# sbatch --account=objectsegvideo --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --output=outputs/%j_baseline_validation.txt --time=00-23:59:59 --partition=booster --signal=TERM@120 --mail-user=ilya.fradlin@rwth-aachen.de --mail-type=BEGIN,END,FAIL scripts/job_submitions/run_on_node.sh

