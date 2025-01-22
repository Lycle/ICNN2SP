#!/bin/bash
#SBATCH --job-name=eval_ef
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=eval_ef.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl

module load scicomp-python-env

srun python3 runner.py --problems cflp_50_50 --eval_ef 1 --n_cpus 16
