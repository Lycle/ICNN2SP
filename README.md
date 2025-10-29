# ICNN2SP
Implementation of the method proposed in the paper "ICNN-enhanced 2SP: Leveraging input convex neural networks for solving two-stage stochastic programming" (https://arxiv.org/abs/2505.05261)

## About
ICNN-enhanced 2SP leverages [input convex neural network (ICNN)](https://proceedings.mlr.press/v70/amos17b.html) to exploit the linear programming (LP) representability in solving two-stage stochastic programming (2SP) problems. The codebase extends Neur2SP’s implementation (https://github.com/khalil-research/Neur2SP) by augmenting it with ICNN-specific functionality.

## Setup and Usage

The pre-generated datasets for benchmark problems are included in the `data` folder. The best neural network parameters found through random search for each problem instance and surrogate model type are stored in `data/nn_params.py`.

### ICNN-enhanced approach
To use the ICNN feature, run the commands below sequentially:
```bash
# Step 1: Train the ICNN surrogate model
python runner.py --problems {PROBLEMS} --train_icnn_e 1

# Step 2: Obtain the best model
python runner.py --problems {PROBLEMS} --get_best_icnn_e_model 1

# Step 3: Reformulate 2SP and obtain solutions
python runner.py --problems {PROBLEMS} --eval_icnn_e 1
```
Replace `{PROBLEMS}` with any of the following problem instances:
- `cflp_10_10`, `cflp_25_25`, `cflp_50_50`
- `sslp_5_25`, `sslp_10_50`, `sslp_15_45`
- `ip_b_E`, `ip_b_H`, `ip_i_E`, `ip_i_H`

### MIP-based approach (from Neur2SP)
To use the MIP-based method to solve 2SP (the original [Neur2SP](https://arxiv.org/abs/2205.12006) approach), simply replace `icnn` with `nn` in the command parameters (i.e., use `--train_nn_e`, `--get_best_nn_e_model`, and `--eval_nn_e`) and follow the same steps as above. This trains a conventional ReLU neural network surrogate using the same dataset as the ICNN-enhanced method, but reformulates the 2SP problem as a mixed-integer program.

### Baseline evaluation
To obtain the baseline results by evaluating extensive form, run:
```bash
python runner.py --problems {PROBLEMS} --eval_ef 1 --n_cpus {N_CPUS}
```