#!/bin/bash
#SBATCH --job-name=train_e_ip
#SBATCH --output=train_e_ip_new.out
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-h100-80g

module load scicomp-python-env
module load cuda/12.2.1

srun python3 runner.py --problems ip_i_E --train_nn_e 1

