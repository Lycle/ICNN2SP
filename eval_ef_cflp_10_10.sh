#!/bin/bash
#SBATCH --job-name=ef_cflp_10_10
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_cflp_10_10_%a.out
#SBATCH --error=eval_ef_cflp_10_10_%a.err
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-7

# Load the required module
module load scicomp-python-env

# Define the commands to run in an array
COMMANDS=(
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 100 --test_set 2 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 100 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 100 --test_set 9 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 500 --test_set 2 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 500 --test_set 3 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 500 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_10_10 --n_scenarios 500 --test_set 9 --n_procs 1"
)

# Get the command for this job array index
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID - 1]}

# Execute the command
echo "Running command: $COMMAND"
eval $COMMAND