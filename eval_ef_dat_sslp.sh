#!/bin/bash
#SBATCH --job-name=ef_sslp
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_sslp_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-120

module load scicomp-python-env

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_ef_sslp.dat | cut -d' ' -f2-)
srun $command