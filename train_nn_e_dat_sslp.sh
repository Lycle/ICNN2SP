#!/bin/bash
#SBATCH --job-name=train_e_sslp
#SBATCH --output=train_e_sslp_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --array=1-3

module load scicomp-python-env
module load cuda/12.2.1

command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" table_train_e_sslp.dat | cut -d' ' -f2-)
srun $command

