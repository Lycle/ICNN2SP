#!/bin/bash
#SBATCH --job-name=ef_sslp_10_50r
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=eval_ef_sslp_10_50r_%a.out
#SBATCH --error=eval_ef_sslp_10_50r_%a.err
#SBATCH --constraint=skl
#SBATCH --partition=batch-skl
#SBATCH --array=1-20

module load scicomp-python-env

# Define the commands to run in an array
COMMANDS=(
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 0 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 1 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 2 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 3 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 4 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 5 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 6 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 8 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 500 --test_set 9 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 0 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 1 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 2 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 3 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 4 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 5 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 6 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 7 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 8 --n_procs 1"
    "python3 -m nsp.scripts.evaluate_extensive --problem sslp_10_50 --n_scenarios 100 --test_set 9 --n_procs 1"
)

# Get the command for this job array index
COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID - 1]}

# Execute the command
echo "Running command: $COMMAND"
eval $COMMAND