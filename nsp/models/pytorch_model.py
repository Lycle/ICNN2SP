import pickle
import time

import numpy as np
import torch
import torch.optim as optim
import wandb
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from torch.utils.data import DataLoader, TensorDataset

from nsp.utils import LossFunction
from .model import Model
from .network import ICNNExpected
from .network import ReLUNetworkExpected

class ICNNExpectedModel(Model):
    def __init__(self,
                 problem,
                 instance,
                 embed_hidden_dim,
                 embed_dim1,
                 embed_dim2,
                 relu_hidden_dim,
                 lr,
                 dropout,
                 optimizer_type,
                 batch_size,
                 loss_fn,
                 wt_lasso,
                 wt_ridge,
                 agg_type,
                 log_freq,
                 n_epochs,
                 use_wandb):

        self.problem = problem
        self.instance = instance

        # model parameters
        self.embed_hidden_dim = embed_hidden_dim
        self.embed_dim1 = embed_dim1
        self.embed_dim2 = embed_dim2
        self.relu_hidden_dim = relu_hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.wt_lasso = wt_lasso
        self.wt_ridge = wt_ridge
        self.log_freq = log_freq
        self.agg_type = agg_type
        self.n_epochs = n_epochs
        self.use_wandb = use_wandb


        self.params = {
            "embed_hidden_dim" : self.embed_hidden_dim,
            "embed_dim1" : self.embed_dim1,
            "embed_dim2" : self.embed_dim2,
            "relu_hidden_dim" : self.relu_hidden_dim,
            "agg_type" : self.agg_type,
            "lr" : self.lr,
            "dropout" : self.dropout,
            "optimizer_type" : self.optimizer_type,
            "batch_size" : self.batch_size,
            "loss_fn" : self.loss_fn,
            "wt_lasso" : self.wt_lasso,
            "wt_ridge" : self.wt_ridge,
            "log_freq" : self.log_freq,
            "n_epochs" : self.n_epochs,
            "use_wandb" :self.use_wandb,
        }


        self.data = None
        self.loader = None
        self.tensors = None  # Initialized in self._get_dataloader()
        self.results = None
        self.is_trained = False

        self.cuda_available = torch.cuda.is_available()
        optimizer_cls = getattr(optim, self.optimizer_type)
        self.criteria = LossFunction(self.loss_fn,
                                     weights={'lasso': self.wt_lasso,
                                              'ridge': self.wt_ridge})

        fs_dim, ss_dim = self._get_feature_dim()
        # initialized as a 'ICNNExpected' model
        self.model = ICNNExpected(
            fs_input_dim=fs_dim,
            ss_input_dim=ss_dim,
            ss_hidden_dim=self.embed_hidden_dim,
            ss_embed_dim1=self.embed_dim1,
            ss_embed_dim2=self.embed_dim2,
            relu_hidden_dim=self.relu_hidden_dim,
            dropout=self.dropout,
            agg_type=self.agg_type,
            bias=False)
        self.model.params = self.params
        self.model = self.model.cuda() if self.cuda_available else self.model
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)

        self.model_clone = ICNNExpected(
            fs_input_dim=fs_dim,
            ss_input_dim=ss_dim,
            ss_hidden_dim=self.embed_hidden_dim,
            ss_embed_dim1=self.embed_dim1,
            ss_embed_dim2=self.embed_dim2,
            relu_hidden_dim=self.relu_hidden_dim,
            dropout=self.dropout,
            agg_type=self.agg_type,
            bias=False)
        self.model_clone = self.model_clone.cuda() if self.cuda_available else self.model_clone
        self.model_clone.eval()

        self.run_name = self.get_run_name()
        if self.use_wandb:
            # initialize wandb
            wandb.init(project=f"nsp_{self.problem}_icnn_e",
                       config={"batch_size": self.batch_size,
                               "lr": self.lr,
                               "dropout": self.dropout,
                               "optimizer": self.optimizer_type,
                               "loss_fn": self.loss_fn,
                               "wt_lasso": self.wt_lasso,
                               "wt_ridge": self.wt_ridge,
                               "log_freq": self.log_freq,
                               "n_epochs": self.n_epochs,
                               "embed_hidden_dim": self.embed_hidden_dim,
                               "embed_dim1": self.embed_dim1,
                               "embed_dim2": self.embed_dim2,
                               "relu_hidden_dim": self.relu_hidden_dim,
                               })
            wandb.run.name = self.run_name

    def train(self, data):
        self.data = data
        self.loader = self._get_dataloader()

        tr_results = {'loss': [], 'mse': [], 'mae': [], 'mape': []}
        val_results = {'mse': [], 'mae': [], 'mape': []}
        best_results = {'mae': np.infty}

        _time = time.time()
        for epoch in range(self.n_epochs):
            ep_loss = self._train_epoch()
            tr_results['loss'].append(np.mean(ep_loss))
            if ((epoch + 1) % self.log_freq) == 0:
                self._val_epoch(epoch, tr_results, val_results, best_results)
        _time = time.time() - _time

        best_results['time'] = _time
        best_results['tr_results'] = tr_results
        best_results['val_results'] = val_results
        self.results = best_results
        self.results['params'] = self.params
        self.is_trained = True

        print('  Finished Training')
        print("  Neural Network train time:", _time)
        if self.use_wandb:
            wandb.log({"time": _time})
            wandb.finish(exit_code=0)

    def get_run_name(self):
        # shorten the run name as it may exceed Windows file path length
        run_name = f"p-{self.problem}_mt-icnn-e-{self.batch_size}_lr-{self.lr}_do-{self.dropout}_" \
                   f"opt-{self.optimizer_type}_loss-{self.loss_fn}_" \
                   f"ed2-{self.embed_dim2}_rhd-{self.relu_hidden_dim}_at-{self.agg_type}"
        return run_name

    def predict(self, *args):
        pass

    def eval_learning_metrics(self):
        pass

    def save_model(self, fp):
        torch.save(self.model_clone, fp)

    def save_results(self, fp):
        with open(fp, 'wb') as p:
            pickle.dump(self.results, p)

    def _train_epoch(self):
        # TODO: Project the hidden layer weights to be non-negative
        # Put the model in training mode
        self.model.train()

        # Train the model and evaluate epoch loss
        ep_loss = []
        for i, data in enumerate(self.loader['tr']):
            x_fs, x_scen, x_n_scen, y = data

            # Variable number of scenarios
            # preds = model(x_fs, x_scen, x_n_scen)
            # Fixed number of scenarios by padding
            preds = self.model(x_fs, x_scen, None)
            loss = self.criteria.get_loss(self.model, preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clamp the weights of relu_output to be non-negative
            self.model.relu_output.weight.data.clamp_(min=0)

            # Verify that weights are non-negative
            if (self.model.relu_output.weight.data < 0).any():
                print("Negative weights detected in relu_output layer!")
            
            ep_loss.append(loss.item())

        if self.use_wandb:
            wandb.log({"loss": np.mean(ep_loss)})

        return ep_loss

    def _val_epoch(self, epoch, tr_results, val_results, best_results):
        self.model.eval()
        with torch.no_grad():
            tr_preds = self.model(self.tensors['x_fs_tr'],
                                  self.tensors['x_scen_tr'],
                                  None).cpu().detach().numpy()
            val_preds = self.model(self.tensors['x_fs_val'],
                                   self.tensors['x_scen_val'],
                                   None).cpu().detach().numpy()

        tr_results['mse'].append(MSE(self.data['y_tr'], tr_preds))
        tr_results['mae'].append(MAE(self.data['y_tr'], tr_preds))
        tr_results['mape'].append(MAPE(self.data['y_tr'], tr_preds))

        val_results['mse'].append(MSE(self.data['y_val'], val_preds))
        val_results['mae'].append(MAE(self.data['y_val'], val_preds))
        val_results['mape'].append(MAPE(self.data['y_val'], val_preds))

        print('  [%d] MSE:   tr: %.3f, val: %.3f' %
              (epoch + 1, tr_results['mse'][-1], val_results['mse'][-1]))
        print('       MAE:   tr: %.3f, val: %.3f' %
              (tr_results['mae'][-1], val_results['mae'][-1]))
        print('       MAPE:  tr: %.3f, val: %.3f' %
              (tr_results['mape'][-1], tr_results['mape'][-1]))

        if val_results['mae'][-1] < best_results['mae']:
            print('    New Best Model')

            self.model_clone.load_state_dict(self.model.state_dict())
            best_results['epoch'] = epoch
            best_results['mae'] = val_results['mae'][-1]
            best_results['tr_mse'] = tr_results['mse'][-1]
            best_results['tr_mae'] = tr_results['mae'][-1]
            best_results['tr_mape'] = tr_results['mape'][-1]
            best_results['val_mse'] = val_results['mse'][-1]
            best_results['val_mae'] = val_results['mae'][-1]
            best_results['val_mape'] = val_results['mape'][-1]

        if self.use_wandb:
            wandb.log({"tr_mse": tr_results['mse'][-1]})
            wandb.log({"tr_mae": tr_results['mae'][-1]})
            wandb.log({"tr_mape": tr_results['mape'][-1]})
            wandb.log({"val_mse": val_results['mse'][-1]})
            wandb.log({"val_mae": val_results['mae'][-1]})
            wandb.log({"val_mape": val_results['mape'][-1]})

    def _get_feature_dim(self):
        """ Get feature dim for initializing neural network. """
        fs_dim, ss_dim = None, None
        if 'cflp' in self.problem:
            fs_dim = self.instance['n_facilities']
            ss_dim = self.instance['n_customers']
        elif 'ip' in self.problem:
            fs_dim = 2
            ss_dim = 2
        elif 'sslp' in self.problem:
            fs_dim = self.instance['n_locations']
            ss_dim = self.instance['n_clients']

        return fs_dim, ss_dim

    def _get_dataloader(self):
        """ For a split (returned from get_data_split), provides features/labels as tensors.  """
        self.data['y_tr'] = self.data['y_tr'].reshape(-1, 1)
        self.data['y_val'] = self.data['y_val'].reshape(-1, 1)

        x_fs_tr = torch.from_numpy(self.data['x_fs_tr']).float()
        x_scen_tr = torch.from_numpy(self.data['x_scen_tr']).float()
        x_n_scen_tr = torch.from_numpy(self.data['x_n_scen_tr']).float()
        y_tr = torch.from_numpy(self.data['y_tr']).float()

        x_fs_val = torch.from_numpy(self.data['x_fs_val']).float()
        x_scen_val = torch.from_numpy(self.data['x_scen_val']).float()
        x_n_scen_val = torch.from_numpy(self.data['x_n_scen_val']).float()
        y_val = torch.from_numpy(self.data['y_val']).float()

        if self.cuda_available:
            x_fs_tr = x_fs_tr.cuda()
            x_scen_tr = x_scen_tr.cuda()
            x_n_scen_tr = x_n_scen_tr.cuda()
            y_tr = y_tr.cuda()
            x_fs_val = x_fs_val.cuda()
            x_scen_val = x_scen_val.cuda()
            x_n_scen_val = x_n_scen_val.cuda()
            y_val = y_val.cuda()

        self.tensors = {'x_fs_tr': x_fs_tr,
                        'x_scen_tr': x_scen_tr,
                        'x_n_scen_tr': x_n_scen_tr,
                        'y_tr': y_tr,
                        'x_fs_val': x_fs_val,
                        'x_scen_val': x_scen_val,
                        'x_n_scen_val': x_n_scen_val,
                        'y_val': y_val}

        # initialize data loaders
        dataset_tr = TensorDataset(x_fs_tr, x_scen_tr, x_n_scen_tr, y_tr)
        dataset_val = TensorDataset(x_fs_val, x_scen_val, x_n_scen_val, y_val)

        loader = {'tr': DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True),
                  'val': DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)}

        return loader

class ReLUNetworkExpectedModel(Model):
    def __init__(self,
                 problem,
                 instance,
                 embed_hidden_dim,
                 embed_dim1,
                 embed_dim2,
                 relu_hidden_dim,
                 lr,
                 dropout,
                 optimizer_type,
                 batch_size,
                 loss_fn,
                 wt_lasso,
                 wt_ridge,
                 agg_type,
                 log_freq,
                 n_epochs,
                 use_wandb):

        self.problem = problem
        self.instance = instance

        # model parameters
        self.embed_hidden_dim = embed_hidden_dim
        self.embed_dim1 = embed_dim1
        self.embed_dim2 = embed_dim2
        self.relu_hidden_dim = relu_hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.wt_lasso = wt_lasso
        self.wt_ridge = wt_ridge
        self.log_freq = log_freq
        self.agg_type = agg_type
        self.n_epochs = n_epochs
        self.use_wandb = use_wandb


        self.params = {
            "embed_hidden_dim" : self.embed_hidden_dim,
            "embed_dim1" : self.embed_dim1,
            "embed_dim2" : self.embed_dim2,
            "relu_hidden_dim" : self.relu_hidden_dim,
            "agg_type" : self.agg_type,
            "lr" : self.lr,
            "dropout" : self.dropout,
            "optimizer_type" : self.optimizer_type,
            "batch_size" : self.batch_size,
            "loss_fn" : self.loss_fn,
            "wt_lasso" : self.wt_lasso,
            "wt_ridge" : self.wt_ridge,
            "log_freq" : self.log_freq,
            "n_epochs" : self.n_epochs,
            "use_wandb" :self.use_wandb,
        }


        self.data = None
        self.loader = None
        self.tensors = None  # Initialized in self._get_dataloader()
        self.results = None
        self.is_trained = False

        self.cuda_available = torch.cuda.is_available()
        optimizer_cls = getattr(optim, self.optimizer_type)
        self.criteria = LossFunction(self.loss_fn,
                                     weights={'lasso': self.wt_lasso,
                                              'ridge': self.wt_ridge})

        fs_dim, ss_dim = self._get_feature_dim()
        self.model = ReLUNetworkExpected(
            fs_input_dim=fs_dim,
            ss_input_dim=ss_dim,
            ss_hidden_dim=self.embed_hidden_dim,
            ss_embed_dim1=self.embed_dim1,
            ss_embed_dim2=self.embed_dim2,
            relu_hidden_dim=self.relu_hidden_dim,
            dropout=self.dropout,
            agg_type=self.agg_type,
            bias=False)
        self.model.params = self.params
        self.model = self.model.cuda() if self.cuda_available else self.model
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr)

        self.model_clone = ReLUNetworkExpected(
            fs_input_dim=fs_dim,
            ss_input_dim=ss_dim,
            ss_hidden_dim=self.embed_hidden_dim,
            ss_embed_dim1=self.embed_dim1,
            ss_embed_dim2=self.embed_dim2,
            relu_hidden_dim=self.relu_hidden_dim,
            dropout=self.dropout,
            agg_type=self.agg_type,
            bias=False)
        self.model_clone = self.model_clone.cuda() if self.cuda_available else self.model_clone
        self.model_clone.eval()

        self.run_name = self.get_run_name()
        if self.use_wandb:
            # initialize wandb
            wandb.init(project=f"nsp_{self.problem}_nn_e",
                       config={"batch_size": self.batch_size,
                               "lr": self.lr,
                               "dropout": self.dropout,
                               "optimizer": self.optimizer_type,
                               "loss_fn": self.loss_fn,
                               "wt_lasso": self.wt_lasso,
                               "wt_ridge": self.wt_ridge,
                               "log_freq": self.log_freq,
                               "n_epochs": self.n_epochs,
                               "embed_hidden_dim": self.embed_hidden_dim,
                               "embed_dim1": self.embed_dim1,
                               "embed_dim2": self.embed_dim2,
                               "relu_hidden_dim": self.relu_hidden_dim,
                               })
            wandb.run.name = self.run_name

    def train(self, data):
        self.data = data
        self.loader = self._get_dataloader()

        tr_results = {'loss': [], 'mse': [], 'mae': [], 'mape': []}
        val_results = {'mse': [], 'mae': [], 'mape': []}
        best_results = {'mae': np.infty}

        _time = time.time()
        for epoch in range(self.n_epochs):
            ep_loss = self._train_epoch()
            tr_results['loss'].append(np.mean(ep_loss))
            if ((epoch + 1) % self.log_freq) == 0:
                self._val_epoch(epoch, tr_results, val_results, best_results)
        _time = time.time() - _time

        best_results['time'] = _time
        best_results['tr_results'] = tr_results
        best_results['val_results'] = val_results
        self.results = best_results
        self.results['params'] = self.params
        self.is_trained = True

        print('  Finished Training')
        print("  Neural Network train time:", _time)
        if self.use_wandb:
            wandb.log({"time": _time})
            wandb.finish(exit_code=0)

    def get_run_name(self):
        # shorten the run name as it may exceed Windows file path length
        run_name = f"p-{self.problem}_mt-nn-e-{self.batch_size}_lr-{self.lr}_do-{self.dropout}_" \
                   f"opt-{self.optimizer_type}_loss-{self.loss_fn}_" \
                   f"ed2-{self.embed_dim2}_rhd-{self.relu_hidden_dim}_at-{self.agg_type}"
        return run_name

    def predict(self, *args):
        pass

    def eval_learning_metrics(self):
        pass

    def save_model(self, fp):
        torch.save(self.model_clone, fp)

    def save_results(self, fp):
        with open(fp, 'wb') as p:
            pickle.dump(self.results, p)

    def _train_epoch(self):
        # Put the model in training mode
        self.model.train()

        # Train the model and evaluate epoch loss
        ep_loss = []
        for i, data in enumerate(self.loader['tr']):
            x_fs, x_scen, x_n_scen, y = data

            # Variable number of scenarios
            # preds = model(x_fs, x_scen, x_n_scen)
            # Fixed number of scenarios by padding
            preds = self.model(x_fs, x_scen, None)
            loss = self.criteria.get_loss(self.model, preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ep_loss.append(loss.item())

        if self.use_wandb:
            wandb.log({"loss": np.mean(ep_loss)})

        return ep_loss

    def _val_epoch(self, epoch, tr_results, val_results, best_results):
        self.model.eval()
        with torch.no_grad():
            tr_preds = self.model(self.tensors['x_fs_tr'],
                                  self.tensors['x_scen_tr'],
                                  None).cpu().detach().numpy()
            val_preds = self.model(self.tensors['x_fs_val'],
                                   self.tensors['x_scen_val'],
                                   None).cpu().detach().numpy()

        tr_results['mse'].append(MSE(self.data['y_tr'], tr_preds))
        tr_results['mae'].append(MAE(self.data['y_tr'], tr_preds))
        tr_results['mape'].append(MAPE(self.data['y_tr'], tr_preds))

        val_results['mse'].append(MSE(self.data['y_val'], val_preds))
        val_results['mae'].append(MAE(self.data['y_val'], val_preds))
        val_results['mape'].append(MAPE(self.data['y_val'], val_preds))

        print('  [%d] MSE:   tr: %.3f, val: %.3f' %
              (epoch + 1, tr_results['mse'][-1], val_results['mse'][-1]))
        print('       MAE:   tr: %.3f, val: %.3f' %
              (tr_results['mae'][-1], val_results['mae'][-1]))
        print('       MAPE:  tr: %.3f, val: %.3f' %
              (tr_results['mape'][-1], tr_results['mape'][-1]))

        if val_results['mae'][-1] < best_results['mae']:
            print('    New Best Model')

            self.model_clone.load_state_dict(self.model.state_dict())
            best_results['epoch'] = epoch
            best_results['mae'] = val_results['mae'][-1]
            best_results['tr_mse'] = tr_results['mse'][-1]
            best_results['tr_mae'] = tr_results['mae'][-1]
            best_results['tr_mape'] = tr_results['mape'][-1]
            best_results['val_mse'] = val_results['mse'][-1]
            best_results['val_mae'] = val_results['mae'][-1]
            best_results['val_mape'] = val_results['mape'][-1]

        if self.use_wandb:
            wandb.log({"tr_mse": tr_results['mse'][-1]})
            wandb.log({"tr_mae": tr_results['mae'][-1]})
            wandb.log({"tr_mape": tr_results['mape'][-1]})
            wandb.log({"val_mse": val_results['mse'][-1]})
            wandb.log({"val_mae": val_results['mae'][-1]})
            wandb.log({"val_mape": val_results['mape'][-1]})

    def _get_feature_dim(self):
        """ Get feature dim for initializing neural network. """
        fs_dim, ss_dim = None, None
        if 'cflp' in self.problem:
            fs_dim = self.instance['n_facilities']
            ss_dim = self.instance['n_customers']
        elif 'ip' in self.problem:
            fs_dim = 2
            ss_dim = 2
        elif 'sslp' in self.problem:
            fs_dim = self.instance['n_locations']
            ss_dim = self.instance['n_clients']

        return fs_dim, ss_dim

    def _get_dataloader(self):
        """ For a split (returned from get_data_split), provides features/labels as tensors.  """
        self.data['y_tr'] = self.data['y_tr'].reshape(-1, 1)
        self.data['y_val'] = self.data['y_val'].reshape(-1, 1)

        x_fs_tr = torch.from_numpy(self.data['x_fs_tr']).float()
        x_scen_tr = torch.from_numpy(self.data['x_scen_tr']).float()
        x_n_scen_tr = torch.from_numpy(self.data['x_n_scen_tr']).float()
        y_tr = torch.from_numpy(self.data['y_tr']).float()

        x_fs_val = torch.from_numpy(self.data['x_fs_val']).float()
        x_scen_val = torch.from_numpy(self.data['x_scen_val']).float()
        x_n_scen_val = torch.from_numpy(self.data['x_n_scen_val']).float()
        y_val = torch.from_numpy(self.data['y_val']).float()

        if self.cuda_available:
            x_fs_tr = x_fs_tr.cuda()
            x_scen_tr = x_scen_tr.cuda()
            x_n_scen_tr = x_n_scen_tr.cuda()
            y_tr = y_tr.cuda()
            x_fs_val = x_fs_val.cuda()
            x_scen_val = x_scen_val.cuda()
            x_n_scen_val = x_n_scen_val.cuda()
            y_val = y_val.cuda()

        self.tensors = {'x_fs_tr': x_fs_tr,
                        'x_scen_tr': x_scen_tr,
                        'x_n_scen_tr': x_n_scen_tr,
                        'y_tr': y_tr,
                        'x_fs_val': x_fs_val,
                        'x_scen_val': x_scen_val,
                        'x_n_scen_val': x_n_scen_val,
                        'y_val': y_val}

        # initialize data loaders
        dataset_tr = TensorDataset(x_fs_tr, x_scen_tr, x_n_scen_tr, y_tr)
        dataset_val = TensorDataset(x_fs_val, x_scen_val, x_n_scen_val, y_val)

        loader = {'tr': DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True),
                  'val': DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)}

        return loader