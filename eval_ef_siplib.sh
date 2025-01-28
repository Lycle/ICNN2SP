#!/bin/bash
#SBATCH --job-name=ef_siplib
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_siplib_%a.out
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-5 

# Load the required module
module load scicomp-python-env

# Define the commands to run in an array
COMMANDS=(
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 'siplib' --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 'siplib' --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 'siplib' --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 1000 --test_set 'siplib' --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 2000 --test_set 'siplib' --n_procs 1"
)

# Get the command for this job array index
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID - 1]}

# Execute the command
echo "Running command: $COMMAND"
eval $COMMAND