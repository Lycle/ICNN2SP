#!/bin/bash
#SBATCH --job-name=ef_sslp_10_50
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_sslp_10_50_%a.out
#SBATCH --error=eval_ef_sslp_10_50_%a.err
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-6 

# Load the required module
module load scicomp-python-env

# Define the commands to run in an array
COMMANDS=(
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 0 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 3 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 4 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 6 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 8 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 50 --test_set 9 --n_procs 1"
)

# Get the command for this job array index
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID - 1]}

# Execute the command
echo "Running command: $COMMAND"
eval $COMMAND