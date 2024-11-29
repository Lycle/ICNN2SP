#!/bin/bash
#SBATCH --job-name=train_e
#SBATCH --output=train_e.out
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1


module load scicomp-python-env
module load cuda/12.2.1

srun python runner.py --problems cflp_50_50 --train_nn_e 1
