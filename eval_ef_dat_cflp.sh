#!/bin/bash
#SBATCH --job-name=ef_cflp
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_cflp_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-99

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_ef_cflp.dat | cut -d' ' -f2-)
srun $command