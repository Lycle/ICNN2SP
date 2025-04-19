import collections

import torch
import torch.nn.functional as F
from torch import nn


class ICNNExpected(nn.Module):

    def __init__(self,
                 fs_input_dim,
                 ss_input_dim,
                 ss_hidden_dim,
                 ss_embed_dim1,
                 ss_embed_dim2,
                 relu_hidden_dim,
                 dropout=0,
                 agg_type="mean",
                 bias=False):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(ICNNExpected, self).__init__()

        # fs: first stage, ss: second stage
        self.fs_input_dim = fs_input_dim

        self.ss_input_dim = ss_input_dim
        self.ss_hidden_dim = ss_hidden_dim
        self.ss_embed_dim1 = ss_embed_dim1
        self.ss_embed_dim2 = ss_embed_dim2

        self.relu_hidden_dim = relu_hidden_dim

        self.dropout = dropout
        self.bias = bias
        self.agg_type = agg_type

        self.output_dim = 1

        # layers for scenario input
        # nn.Linear: to create a fully connected (dense) layer
        self.scen_input = nn.Linear(self.ss_input_dim, self.ss_hidden_dim, bias=self.bias)
        self.scen_embed1 = nn.Linear(self.ss_hidden_dim, self.ss_embed_dim1, bias=self.bias)
        self.scen_embed2 = nn.Linear(self.ss_embed_dim1, self.ss_embed_dim2)

        # for relu layer
        self.relu_input = nn.Linear(self.fs_input_dim + self.ss_embed_dim2, self.relu_hidden_dim)
        self.relu_output = nn.Linear(self.relu_hidden_dim, self.output_dim)
        self.skip = nn.Linear(self.fs_input_dim + self.ss_embed_dim2, self.output_dim, bias=False)  # skip connection
        

    def forward(self, x_fs, x_scen, x_n_scen=None):
        """ Forward pass. """

        # embed scenarios
        x_scen_embed = self.embed_scenarios(x_scen, x_n_scen)

        # concat first stage solution and scenario embedding
        concat = torch.cat((x_fs, x_scen_embed), 1)

        # get aggregate prediction of ReLU path
        relu_hidden = self.relu_input(concat)
        relu_hidden = F.relu(relu_hidden)
        if self.dropout:
            relu_hidden = F.dropout(relu_hidden, p=self.dropout)

        relu_hidden = self.relu_output(relu_hidden)

        # skip connection path
        skip_out = self.skip(concat)

        # Final output by summing both paths
        output = relu_hidden + skip_out

        return output

    def embed_scenarios(self, x_scen, x_n_scen=None):
        """ Get scenario embedding
        """
        #  transform the scenario inputs (x_scen) into lower-dimensional embeddings

        # for each batch, pass non-padded values in. 
        if x_n_scen is not None:
            x_scen_embed = []
            for i in range(x_scen.shape[0]):
                n_scen = int(x_n_scen[i].item())
                x_scen_in = x_scen[i, :n_scen]
                x_scen_in = torch.reshape(x_scen_in, (1, x_scen_in.shape[0], x_scen_in.shape[1]))

                # embed 
                x_scen_embed_i = self.scen_input(x_scen_in)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed_i = self.scen_embed1(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                if self.agg_type == "sum":
                    x_scen_embed_i = torch.sum(x_scen_embed_i, axis=1)  # sum all inputs
                elif self.agg_type == "mean":
                    x_scen_embed_i = torch.mean(x_scen_embed_i, axis=1)  # mean all inputs

                x_scen_embed_i = self.scen_embed2(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed.append(x_scen_embed_i)

            x_scen_embed = torch.stack(x_scen_embed)
            x_scen_embed = torch.reshape(x_scen_embed, (x_scen_embed.shape[0], x_scen_embed.shape[2]))
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        # assume no padding, i.e. full scenario set
        else:
            x_scen_embed = self.scen_input(x_scen)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            x_scen_embed = self.scen_embed1(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            if self.agg_type == "sum":
                x_scen_embed = torch.sum(x_scen_embed, axis=1)  # sum all inputs
            elif self.agg_type == "mean":
                x_scen_embed = torch.mean(x_scen_embed, axis=1)  # mean all inputs

            x_scen_embed = self.scen_embed2(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        return x_scen_embed

class ReLUNetworkExpected(nn.Module):

    def __init__(self,
                 fs_input_dim,
                 ss_input_dim,
                 ss_hidden_dim,
                 ss_embed_dim1,
                 ss_embed_dim2,
                 relu_hidden_dim,
                 dropout=0,
                 agg_type="mean",
                 bias=False):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(ReLUNetworkExpected, self).__init__()

        self.fs_input_dim = fs_input_dim

        self.ss_input_dim = ss_input_dim
        self.ss_hidden_dim = ss_hidden_dim
        self.ss_embed_dim1 = ss_embed_dim1
        self.ss_embed_dim2 = ss_embed_dim2

        self.relu_hidden_dim = relu_hidden_dim

        self.dropout = dropout
        self.bias = bias
        self.agg_type = agg_type

        self.output_dim = 1

        # layers for scenario input
        self.scen_input = nn.Linear(self.ss_input_dim, self.ss_hidden_dim, bias=self.bias)
        self.scen_embed1 = nn.Linear(self.ss_hidden_dim, self.ss_embed_dim1, bias=self.bias)
        self.scen_embed2 = nn.Linear(self.ss_embed_dim1, self.ss_embed_dim2)

        # for relu layer
        self.relu_input = nn.Linear(self.fs_input_dim + self.ss_embed_dim2, self.relu_hidden_dim)
        self.relu_output = nn.Linear(self.relu_hidden_dim, self.output_dim)

    def forward(self, x_fs, x_scen, x_n_scen=None):
        """ Forward pass. """

        # embed scenarios
        x_scen_embed = self.embed_scenarios(x_scen, x_n_scen)

        # concat first stage solution and scenario embedding
        x = torch.cat((x_fs, x_scen_embed), 1)

        # get aggregate prediction
        x = self.relu_input(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout)

        x = self.relu_output(x)

        return x

    def embed_scenarios(self, x_scen, x_n_scen=None):
        """ Get scenario embedding
        """
        # for each batch, pass-non padded values in. 
        if x_n_scen is not None:
            x_scen_embed = []
            for i in range(x_scen.shape[0]):
                n_scen = int(x_n_scen[i].item())
                x_scen_in = x_scen[i, :n_scen]
                x_scen_in = torch.reshape(x_scen_in, (1, x_scen_in.shape[0], x_scen_in.shape[1]))

                # embed 
                x_scen_embed_i = self.scen_input(x_scen_in)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed_i = self.scen_embed1(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                if self.agg_type == "sum":
                    x_scen_embed_i = torch.sum(x_scen_embed_i, axis=1)  # sum all inputs
                elif self.agg_type == "mean":
                    x_scen_embed_i = torch.mean(x_scen_embed_i, axis=1)  # mean all inputs

                x_scen_embed_i = self.scen_embed2(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed.append(x_scen_embed_i)

            x_scen_embed = torch.stack(x_scen_embed)
            x_scen_embed = torch.reshape(x_scen_embed, (x_scen_embed.shape[0], x_scen_embed.shape[2]))
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        # assume no padding, i.e. full scenario set
        else:
            x_scen_embed = self.scen_input(x_scen)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            x_scen_embed = self.scen_embed1(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            if self.agg_type == "sum":
                x_scen_embed = torch.sum(x_scen_embed, axis=1)  # sum all inputs
            elif self.agg_type == "mean":
                x_scen_embed = torch.mean(x_scen_embed, axis=1)  # mean all inputs

            x_scen_embed = self.scen_embed2(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        return x_scen_embed