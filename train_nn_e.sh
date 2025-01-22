#!/bin/bash
#SBATCH --job-name=train_e_cflp_25_25
#SBATCH --output=train_e_cflp_25_25.out
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-h100-80g

module load scicomp-python-env
module load cuda/12.2.1

srun python runner.py --problems cflp_25_25 --train_nn_e 1

