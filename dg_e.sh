#!/bin/bash
#SBATCH --job-name=run_dg_e
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=dg_e.out

module load scicomp-python-env

srun python3 runner.py --problems cflp_50_50 --run_dg_e 1 --n_cpus 16
