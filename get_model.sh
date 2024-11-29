#!/bin/bash
#SBATCH --job-name=run_dg_e
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=get_model.out

module load scicomp-python-env

srun python3 runner.py --problems cflp_50_50 --get_best_nn_e_model 1
