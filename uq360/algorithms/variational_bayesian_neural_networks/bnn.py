import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from uq360.algorithms.builtinuq import BuiltinUQ
from uq360.models.bayesian_neural_networks.bnn_models import horseshoe_mlp, bayesian_mlp


class BnnRegression(BuiltinUQ):
    """
    Variationally trained BNNs with Gaussian and Horseshoe [6]_ priors for regression.

    References:
        .. [6] Ghosh, Soumya, Jiayu Yao, and Finale Doshi-Velez. "Structured variational learning of Bayesian neural
            networks with horseshoe priors." International Conference on Machine Learning. PMLR, 2018.
    """
    def __init__(self,  config, prior="Gaussian"):
        """

        Args:
            config: a dictionary specifying network and learning hyperparameters.
            prior: BNN priors specified as a string. Supported priors are Gaussian, Hshoe, RegHshoe
        """
        super(BnnRegression, self).__init__()
        self.config = config
        if prior == "Gaussian":
            self.net = bayesian_mlp.BayesianRegressionNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                          num_nodes=config['num_nodes'], num_layers=config['num_layers'])
            self.config['use_reg_hshoe'] = None
        elif prior == "Hshoe":
            self.net = horseshoe_mlp.HshoeRegressionNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                          num_nodes=config['num_nodes'], num_layers=config['num_layers'],
                                                       hshoe_scale=config['hshoe_scale'])
            self.config['use_reg_hshoe'] = False
        elif prior == "RegHshoe":
            self.net = horseshoe_mlp.HshoeRegressionNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                                   num_nodes=config['num_nodes'], num_layers=config['num_layers'],
                                                   hshoe_scale=config['hshoe_scale'],
                                                   use_reg_hshoe=config['use_reg_hshoe'])
            self.config['use_reg_hshoe'] = True
        else:
            raise NotImplementedError("'prior' must be a string. It can be one of Gaussian, Hshoe, RegHshoe")

    def get_params(self, deep=True):
        return {"prior": self.prior, "config": self.config}

    def fit(self, X, y):
        """ Fit the BNN regression model.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """
        torch.manual_seed(1234)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['step_size'])
        neg_elbo = torch.zeros([self.config['num_epochs'], 1])
        params_store = {}
        for epoch in range(self.config['num_epochs']):
            loss = self.net.neg_elbo(num_batches=1, x=X, y=y.float().unsqueeze(dim=1)) / X.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hasattr(self.net, 'fixed_point_updates'):
                # for hshoe or regularized hshoe nets
                self.net.fixed_point_updates()
            neg_elbo[epoch] = loss.item()
            if (epoch + 1) % 10 == 0:
                # print ((net.noise_layer.bhat/net.noise_layer.ahat).data.numpy()[0])
                print('Epoch[{}/{}], neg elbo: {:.6f}, noise var: {:.6f}'
                      .format(epoch + 1, self.config['num_epochs'], neg_elbo[epoch].item() / X.shape[0],
                              self.net.get_noise_var()))
            params_store[epoch] = copy.deepcopy(self.net.state_dict()) # for small nets we can just store all.
        best_model_id = neg_elbo.argmin()  # loss_val_store.argmin() #
        self.net.load_state_dict(params_store[best_model_id.item()])

        return self

    def predict(self, X, mc_samples=100, return_dists=False, return_epistemic=True, return_epistemic_dists=False):
        """
        Obtain predictions for the test points.

        In addition to the mean and lower/upper bounds, also returns epistemic uncertainty (return_epistemic=True)
        and full predictive distribution (return_dists=True).

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            mc_samples: Number of Monte-Carlo samples.
            return_dists: If True, the predictive distribution for each instance using scipy distributions is returned.
            return_epistemic: if True, the epistemic upper and lower bounds are returned.
            return_epistemic_dists: If True, the epistemic distribution for each instance using scipy distributions
                is returned.

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
            y_lower_epistemic: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of epistemic component of the predictive distribution of the test points.
                Only returned when `return_epistemic` is True.
            y_upper_epistemic: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of epistemic component of the predictive distribution of the test points.
                Only returned when `return_epistemic` is True.
            dists: list of predictive distribution as `scipy.stats` objects with length n_samples.
                Only returned when `return_dists` is True.
        """
        epistemic_out = np.zeros([mc_samples, X.shape[0]])
        total_out = np.zeros([mc_samples, X.shape[0]])
        for s in np.arange(mc_samples):
            pred = self.net(X).data.numpy().ravel()
            epistemic_out[s] = pred
            total_out[s] = pred + np.sqrt(self.net.get_noise_var()) * np.random.randn(pred.shape[0])
        y_total_std = np.std(total_out, axis=0)
        y_epi_std = np.std(epistemic_out, axis=0)
        y_mean = np.mean(total_out, axis=0)
        y_lower = y_mean - 2 * y_total_std
        y_upper = y_mean + 2 * y_total_std
        y_epi_lower = y_mean - 2 * y_epi_std
        y_epi_upper = y_mean + 2 * y_epi_std

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        if return_epistemic:
            Result = namedtuple('res', Result._fields + ('lower_epistemic', 'upper_epistemic',))
            res = Result(*res, lower_epistemic=y_epi_lower, upper_epistemic=y_epi_upper)

        if return_dists:
            dists = [norm(loc=y_mean[i], scale=y_total_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_dists',))
            res = Result(*res, y_dists=dists)

        if return_epistemic_dists:
            epi_dists = [norm(loc=y_mean[i], scale=y_epi_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_epistemic_dists',))
            res = Result(*res, y_epistemic_dists=epi_dists)

        return res


class BnnClassification(BuiltinUQ):
    """
    Variationally trained BNNs with Gaussian and Horseshoe [6]_ priors for classification.
    """
    def __init__(self,  config, prior="Gaussian", device=None):
        """

        Args:
            config: a dictionary specifying network and learning hyperparameters.
            prior: BNN priors specified as a string. Supported priors are Gaussian, Hshoe, RegHshoe
        """
        super(BnnClassification, self).__init__()
        self.config = config
        self.device = device
        if prior == "Gaussian":
            self.net = bayesian_mlp.BayesianClassificationNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                          num_nodes=config['num_nodes'], num_layers=config['num_layers'])
            self.config['use_reg_hshoe'] = None
        elif prior == "Hshoe":
            self.net = horseshoe_mlp.HshoeClassificationNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                          num_nodes=config['num_nodes'], num_layers=config['num_layers'],
                                                       hshoe_scale=config['hshoe_scale'])
            self.config['use_reg_hshoe'] = False
        elif prior == "RegHshoe":
            self.net = horseshoe_mlp.HshoeClassificationNet(ip_dim=config['ip_dim'], op_dim=config['op_dim'],
                                                   num_nodes=config['num_nodes'], num_layers=config['num_layers'],
                                                   hshoe_scale=config['hshoe_scale'],
                                                   use_reg_hshoe=config['use_reg_hshoe'])
            self.config['use_reg_hshoe'] = True
        else:
            raise NotImplementedError("'prior' must be a string. It can be one of Gaussian, Hshoe, RegHshoe")
        if "batch_size" not in self.config:
            self.config["batch_size"] = 50
        self.net = self.net.to(device)

    def get_params(self, deep=True):
        return {"prior": self.prior, "config": self.config, "device": self.device}

    def fit(self, X=None, y=None, train_loader=None):
        """ Fits BNN regression model.

        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples, n_classes).
                Features vectors of the training data or the probability scores from the base model.
                Ignored if train_loader is not None.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values
                Ignored if train_loader is not None.
            train_loader: pytorch train_loader object.

        Returns:
            self

        """
        if train_loader is None:
            train = data_utils.TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
            train_loader = data_utils.DataLoader(train, batch_size=self.config['batch_size'], shuffle=True)

        torch.manual_seed(1234)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['step_size'])
        neg_elbo = torch.zeros([self.config['num_epochs'], 1])
        params_store = {}
        for epoch in range(self.config['num_epochs']):
            avg_loss = 0.0
            for batch_x, batch_y in train_loader:
                loss = self.net.neg_elbo(num_batches=len(train_loader), x=batch_x, y=batch_y) / batch_x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if hasattr(self.net, 'fixed_point_updates'):
                    # for hshoe or regularized hshoe nets
                    self.net.fixed_point_updates()

                avg_loss += loss.item()

            neg_elbo[epoch] = avg_loss / len(train_loader)

            if (epoch + 1) % 10 == 0:
                # print ((net.noise_layer.bhat/net.noise_layer.ahat).data.numpy()[0])
                print('Epoch[{}/{}], neg elbo: {:.6f}'
                      .format(epoch + 1, self.config['num_epochs'], neg_elbo[epoch].item()))
            params_store[epoch] = copy.deepcopy(self.net.state_dict()) # for small nets we can just store all.
        best_model_id = neg_elbo.argmin()  # loss_val_store.argmin() #
        self.net.load_state_dict(params_store[best_model_id.item()])

        return self

    def predict(self, X, mc_samples=100):
        """
        Obtain calibrated predictions for the test points.

        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples, n_classes).
                Features vectors of the training data or the probability scores from the base model.
            mc_samples: Number of Monte-Carlo samples.

        Returns:
            namedtuple: A namedtupe that holds

            y_pred: ndarray of shape (n_samples,)
                Predicted labels of the test points.
            y_prob: ndarray of shape (n_samples, n_classes)
                Predicted probability scores of the classes.
            y_prob_var: ndarray of shape (n_samples,)
                Variance of the prediction on the test points.
            y_prob_samples: ndarray of shape (mc_samples, n_samples, n_classes)
                Samples from the predictive distribution.

        """

        X = torch.Tensor(X)
        y_prob_samples = [F.softmax(self.net(X), dim=1).detach().numpy() for _ in np.arange(mc_samples)]

        y_prob_samples_stacked = np.stack(y_prob_samples)
        prob_mean = np.mean(y_prob_samples_stacked, 0)
        prob_var = np.std(y_prob_samples_stacked, 0) ** 2

        if len(np.shape(prob_mean)) == 1:
            y_pred_labels = prob_mean > 0.5

        else:
            y_pred_labels = np.argmax(prob_mean, axis=1)

        Result = namedtuple('res', ['y_pred', 'y_prob', 'y_prob_var', 'y_prob_samples'])
        res = Result(y_pred_labels, prob_mean, prob_var, y_prob_samples)

        return res
