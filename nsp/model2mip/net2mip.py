import gurobipy as gp
import numpy as np


class Net2MIPPerScenario(object):
    """
        Take a learned neural representation of the Q(x, k) and fuse
        it into the parent representation

    Params
    ------

    """

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 network,
                 scenario_representations,
                 scenario_probs=None,
                 M_plus=1e5,
                 M_minus=1e5):

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.network = network
        self.scenario_representations = scenario_representations
        self.scenario_probs = scenario_probs
        self.M_plus = M_plus
        self.M_minus = M_minus

        # if no scenario probabilities are given, assume equal probability
        self.n_scenarios = len(self.scenario_representations)
        if self.scenario_probs is None:  # 'sulfur' in scenario_representations:
            self.scenario_probs = np.ones(self.n_scenarios) / self.n_scenarios

        self.Q_var_lst = []
        self.scenario_index = 0

    def get_mip(self):
        """ Gets MIP embedding of NN. """
        for scenario_prob, scenario in zip(self.scenario_probs,
                                           self.scenario_representations):
            Q_var = self._add_scenario_to_mip(scenario)
            Q_var.setAttr("obj", scenario_prob)
            self.Q_var_lst.append(Q_var)
            self.scenario_index += 1

        self.gp_model.update()

        return self.gp_model

    def _add_scenario_to_mip(self, scenario):
        """
        Take a learned neural representation of the Q(x, k) and fuse
        it into the parent representation.
        """

        nVar = len(self.gp_vars.keys())

        # Extract learned weights and bias from the neural network
        W, B = [], []
        for name, param in self.network.named_parameters():
            if 'weight' in name:
                W.append(param.cpu().detach().numpy())
            if 'bias' in name:
                B.append(param.cpu().detach().numpy())

        XX = []
        for k, (wt, b) in enumerate(zip(W, B)):

            outSz, inpSz = wt.shape

            X, S, Z = [], [], []
            for j in range(outSz):
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'
                s_name = f's_{self.scenario_index}_{k + 1}_{j}'
                z_name = f'z_{self.scenario_index}_{k + 1}_{j}'

                if k < len(W) - 1:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                    S.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=s_name))
                    Z.append(self.gp_model.addVar(vtype=gp.GRB.BINARY, name=z_name))
                else:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name=x_name))

                # W out-by-in
                # x in-by-1
                # _eq = W . x
                _eq = 0
                for i in range(inpSz):
                    # First layer weights are partially multiplied by gp.var and features
                    if k == 0:
                        # Multiply gp vars
                        if i < nVar:
                            _eq += wt[j][i] * self.gp_vars[i]
                        else:
                            _eq += wt[j][i] * scenario[i - nVar]
                    else:
                        _eq += wt[j][i] * XX[-1][i]

                # Add bias
                _eq += b[j]

                # Add constraint for each output neuron
                if k < len(W) - 1:
                    self.gp_model.addConstr(_eq == X[-1] - S[-1], name=f"mult_{x_name}__{s_name}")
                    self.gp_model.addConstr(X[-1] <= self.M_plus * (1 - Z[-1]), name=f"bm_{x_name}")
                    self.gp_model.addConstr(S[-1] <= self.M_minus * (Z[-1]), name=f"bm_{s_name}")

                else:
                    self.gp_model.addConstr(_eq == X[-1], name=f"mult_out_{x_name}__{s_name}")

                # Save current layers gurobi vars
                XX.append(X)

        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var


class Net2MIPExpected(object):
    """
        Take a learned neural representation of the sum_k Q(x, k) and fuse 
        it into the parent representation

    Params
    ------

    """
    #TODO: Add feature to transfer ICNN to LP

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 network,
                 scenario_embedding,
                 scenario_probs=None):
                 

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.network = network
        self.scenario_embedding = scenario_embedding

        self.Q_var_lst = []
        self.scenario_index = 0

        self.extract_weights()

    def get_mip(self):
        """ Gets MIP embedding of NN. """
        self._create_lp()
        return self.gp_model

    def extract_weights(self):
        """ Extract weights from model.  """
        # Extract weights and biases from the neural network, including skip connections
        self.W_fc = [self.network.relu_input.weight.cpu().detach().numpy(),
                  self.network.relu_output.weight.cpu().detach().numpy()]
        self.B_fc = [self.network.relu_input.bias.cpu().detach().numpy(),
                  self.network.relu_output.bias.cpu().detach().numpy()]
        self.W_skip = [None, self.network.skip.weight.cpu().detach().numpy()]
        

    def _create_lp_old(self):
        """
        Convert the ICNN part of the neural network into LP constraints and add them to the Gurobi model.
        """
        z = {}
        # Initialize z[0] with input variables (concatenation of first-stage variables and scenario embeddings)
        z[0] = []
        for var in self.gp_vars:
            z[0].append(var)
        for var in self.scenario_embedding:
            z[0].append(var)

        num_layers = len(self.W_fc)
        
        for k in range(1, num_layers + 1):
            W_fc = self.W_fc[k - 1]       # Weights of the k-th layer
            b_fc = self.B_fc[k - 1]       # Biases of the k-th layer
            W_skip = self.W_skip[k - 1]   # Skip weights (None for the first layer)
            
            num_neurons = W_fc.shape[0]
            z[k] = []

            for j in range(num_neurons):
                var_name = f'z_{self.scenario_index}_{k}_{j}'

                # Create variables for the neurons
                if k == num_layers:
                    # Output layer variable (can be negative)
                    z_var = self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name=var_name)
                else:
                    # Hidden layer variables (ReLU activations, non-negative)
                    z_var = self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=var_name)
                z[k].append(z_var)

                # Build the affine expression for the neuron
                expr = b_fc[j]

                # Contribution from the previous layer
                for i in range(len(z[k - 1])):
                    expr += W_fc[j][i] * z[k - 1][i]

                # Contribution from the skip connection (if any)
                if W_skip is not None:
                    for i in range(len(z[0])):
                        expr += W_skip[j][i] * z[0][i]

                # Add the constraint: z_var >= expr
                self.gp_model.addConstr(z_var >= expr, name=f'icnn_{var_name}')

                # Enforce non-negativity for hidden layers (ReLU activation)
                if k < num_layers:
                    self.gp_model.addConstr(z_var >= 0, name=f'nonneg_{var_name}')

        # Get the output variable from the last layer
        self.Q_var = z[num_layers][0]  # Assuming output is scalar

        # ======================
        # Modify the LP's objective to include the ICNN's output as a penalty term
        # ======================
        current_obj = self.gp_model.getObjective()

        if current_obj is not None:
            if self.gp_model.ModelSense == gp.GRB.MINIMIZE:
                # If the model is minimizing, add Q_var as a penalty
                new_obj = gp.LinExpr(current_obj)
                new_obj += self.Q_var  # Coefficient of 1 for Q_var; adjust as needed
                self.gp_model.setObjective(new_obj, gp.GRB.MINIMIZE)
                
            else:
                # If the model is maximizing, subtract Q_var as a penalty
                new_obj = gp.LinExpr(current_obj)
                new_obj -= self.Q_var  # Coefficient of -1 for Q_var; adjust as needed
                self.gp_model.setObjective(new_obj, gp.GRB.MAXIMIZE)
        else:
            # If no objective is set, set Q_var as the objective
            self.gp_model.setObjective(self.Q_var, gp.GRB.MINIMIZE)


        # Update the model to integrate new variables and constraints
        self.gp_model.update()
        for constr in self.gp_model.getConstrs():
            print(f"{constr.ConstrName}: {self.gp_model.getRow(constr)}")

    def _create_lp(self):
        nVar = len(self.gp_vars.keys())     # 16
        # print(self.scenario_embedding)
        XX = []
        for k, (wt, b, wt_skip) in enumerate(zip(self.W_fc, self.B_fc, self.W_skip)):
            # 128 -> 24 -> 1
            outSz, inpSz = wt.shape

            X = []
            for j in range(outSz):     # 1st:128; 2nd layer:1
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'
                # print(f"x_name: {x_name}")

                if k < len(self.W_fc) - 1:  # len(self.W_fc) = 2
                    # X >= 0
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                else:
                    # set the penalty term coefficient 
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, obj=1, name=x_name))
                
                _eq = 0
                for i in range(inpSz):    # 1st: 24; 2nd: 128
                    # First layer weights are partially multiplied by gp.var and features
                    if k == 0:
                        # Multiply gp vars
                        if i < nVar:
                            _eq += wt[j][i] * self.gp_vars[i]
                        else:
                            _eq += wt[j][i] * self.scenario_embedding[i - nVar]
                    else:
                        _eq += wt[j][i] * XX[-1][i]
                        if i < wt_skip.shape[1]:
                            if i < nVar:
                                _eq += wt_skip[j][i] * self.gp_vars[i]
                            else:
                                _eq += wt_skip[j][i] * self.scenario_embedding[i - nVar]
                        
                # Add bias
                _eq += b[j]

                # Add constraint for each output neuron
                if k < len(self.W_fc) - 1:
                    self.gp_model.addConstr(_eq <= X[-1], name=f"mult_{x_name}")
                    # self.gp_model.update()
                    # print(f"Added constraint for hidden layer {k}, neuron {j}: {_eq} <= {X[-1]}\n")
                else:
                    self.gp_model.addConstr(_eq <= X[-1], name=f"mult_out_{x_name}")
                    # self.gp_model.update()
                    # print(f"Added constraint for output layer {k}, neuron {j}: {_eq} <= {X[-1]}\n")

                # Save current layers gurobi vars
                XX.append(X)
        
        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var

    def _create_mip(self):
        """
        Take a learned neural representation of the Q(x, k) and fuse 
        it into the parent representation.
        """

        nVar = len(self.gp_vars.keys())

        XX = []
        for k, (wt, b) in enumerate(zip(self.W, self.B)):

            outSz, inpSz = wt.shape

            X, S, Z = [], [], []
            for j in range(outSz):
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'
                s_name = f's_{self.scenario_index}_{k + 1}_{j}'
                z_name = f'z_{self.scenario_index}_{k + 1}_{j}'

                if k < len(self.W) - 1:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                    S.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=s_name))
                    Z.append(self.gp_model.addVar(vtype=gp.GRB.BINARY, name=z_name))
                else:   # Last layer
                    # obj=1 sets the coefficient of this variable in the objective function to 1
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, obj=1, name=x_name))

                # W out-by-in
                # x in-by-1
                # _eq = W . x 
                _eq = 0
                for i in range(inpSz):
                    # First layer weights are partially multiplied by gp.var and features
                    if k == 0:
                        # Multiply gp vars
                        if i < nVar:
                            _eq += wt[j][i] * self.gp_vars[i]
                        else:
                            _eq += wt[j][i] * self.scenario_embedding[i - nVar]
                    else:
                        _eq += wt[j][i] * XX[-1][i]

                # Add bias
                _eq += b[j]

                # Add constraint for each output neuron 
                if k < len(self.W) - 1:
                    self.gp_model.addConstr(_eq == X[-1] - S[-1], name=f"mult_{x_name}__{s_name}")
                    self.gp_model.addConstr(X[-1] <= self.M_plus * (1 - Z[-1]), name=f"bm_{x_name}")
                    self.gp_model.addConstr(S[-1] <= self.M_minus * (Z[-1]), name=f"bm_{s_name}")

                else:
                    self.gp_model.addConstr(_eq == X[-1], name=f"mult_out_{x_name}__{s_name}")

                # Save current layers gurobi vars
                XX.append(X)

        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var
