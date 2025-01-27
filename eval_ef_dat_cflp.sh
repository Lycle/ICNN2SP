#!/bin/bash
#SBATCH --job-name=eval_ef_cflp_10
#SBATCH --time=3:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_cflp_10_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-11

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_ef_cflp_10_10.dat | cut -d' ' -f2-)
srun $command