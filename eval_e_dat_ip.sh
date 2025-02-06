#!/bin/bash
#SBATCH --job-name=val_e_ip
#SBATCH --time=00:05:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=eval_e_ip_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=3

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_eval_e_ip.dat | cut -d' ' -f2-)
srun $command