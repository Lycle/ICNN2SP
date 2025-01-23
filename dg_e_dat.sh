#!/bin/bash
#SBATCH --job-name=run_dg_e
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=dg_e_%A_%a.out
#SBATCH --array=1-4

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_dg_e.dat | cut -d' ' -f2-)
srun $command
