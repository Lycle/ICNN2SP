#!/bin/bash
#SBATCH --job-name=eval_e
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=eval_e_cflp_10_10.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl

module load scicomp-python-env

srun python3 runner.py --problems cflp_10_10 --eval_nn_e 1 --n_cpus 16

