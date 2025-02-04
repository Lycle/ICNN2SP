#!/bin/bash
#SBATCH --job-name=dg_e_i_E
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=dg_e_ip_i_E_new.out

module load scicomp-python-env

srun python3 runner.py --problems ip_i_E --run_dg_e 1 --n_cpus 16
