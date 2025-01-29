#!/bin/bash
#SBATCH --job-name=ef_cflp_25_25
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=eval_ef_cflp_25_25_%a.out
#SBATCH --error=eval_ef_cflp_25_25_%a.err
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-9

# Load the required module
module load scicomp-python-env

# Define the commands to run in an array
COMMANDS=(
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 100 --test_set 1 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 100 --test_set 2 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 100 --test_set 5 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 100 --test_set 6 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 100 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 1000 --test_set 1 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 1000 --test_set 3 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 1000 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem cflp_25_25 --n_scenarios 1000 --test_set 8 --n_procs 1"
)

# Get the command for this job array index
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID - 1]}

# Execute the command
echo "Running command: $COMMAND"
eval $COMMAND