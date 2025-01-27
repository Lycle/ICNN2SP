#!/bin/bash
#SBATCH --job-name=ef_cflp_25_50
#SBATCH --time=3:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_cflp_25_50_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-66

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_ef_cflp_25_50.dat | cut -d' ' -f2-)
srun $command