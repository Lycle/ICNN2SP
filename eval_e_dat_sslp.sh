#!/bin/bash
#SBATCH --job-name=val_e_sslp
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=eval_e_sslp_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-3

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_eval_e_sslp.dat | cut -d' ' -f2-)
srun $command