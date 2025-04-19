import gurobipy as gp
import numpy as np


class Net2LPExpected(object):
    
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
        """ Gets LP embedding of ICNN. """
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
        
    def _create_lp(self):
        nVar = len(self.gp_vars.keys())
        XX = []
        for k, (wt, b, wt_skip) in enumerate(zip(self.W_fc, self.B_fc, self.W_skip)):
            outSz, inpSz = wt.shape

            X = []
            for j in range(outSz):
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'

                if k < len(self.W_fc) - 1:
                    # X >= 0
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                else:
                    # set the penalty term coefficient 
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, obj=1, name=x_name))
                
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
                else:
                    self.gp_model.addConstr(_eq <= X[-1], name=f"mult_out_{x_name}")

                # Save current layers gurobi vars
                XX.append(X)
        
        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var

class Net2MIPExpected(object):

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 network,
                 scenario_embedding,
                 scenario_probs=None,
                 M_plus=1e5,
                 M_minus=1e5):

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.network = network
        self.scenario_embedding = scenario_embedding
        self.M_plus = M_plus
        self.M_minus = M_minus

        self.Q_var_lst = []
        self.scenario_index = 0

        self.extract_weights()

    def get_mip(self):
        """ Gets MIP embedding of NN. """
        self._create_mip()
        return self.gp_model

    def extract_weights(self):
        """ Extract weights from model.  """
        # Extract learned weights and bias from the neural network

        self.W = [self.network.relu_input.weight.cpu().detach().numpy(),
                  self.network.relu_output.weight.cpu().detach().numpy()]
        self.B = [self.network.relu_input.bias.cpu().detach().numpy(),
                  self.network.relu_output.bias.cpu().detach().numpy()]

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
                else:
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