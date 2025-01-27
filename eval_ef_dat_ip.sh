#!/bin/bash
#SBATCH --job-name=ef_ip
#SBATCH --time=3:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_ip_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-28

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_ef_ip.dat | cut -d' ' -f2-)
srun $command