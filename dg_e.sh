#!/bin/bash
#SBATCH --job-name=run_dg_e
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=dg_e_sslp_10_50.out

module load scicomp-python-env

srun python3 runner.py --problems sslp_10_50 --run_dg_e 1 --n_cpus 16
