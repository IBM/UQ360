from collections import namedtuple

import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from uq360.algorithms.builtinuq import BuiltinUQ
from uq360.models.heteroscedastic_mlp import GaussianNoiseMLPNet as _MLPNet

np.random.seed(42)
torch.manual_seed(42)

class HeteroscedasticRegression(BuiltinUQ):
    """ Wrapper for heteroscedastic regression. We learn to predict targets given features, 
         assuming that the targets are noisy and that the amount of noise varies between data points. 
         https://en.wikipedia.org/wiki/Heteroscedasticity
    """

    def __init__(self, model_type=None, model=None, config=None, device=None, verbose=True):
        """
        Args:
            model_type: The base model architecture. Currently supported values are [mlp].
                mlp modeltype learns a multi-layer perceptron with a heteroscedastic Gaussian likelihood. Both the
                mean and variance of the Gaussian are functions of the data point ->git  N(y_n | mlp_mu(x_n), mlp_var(x_n))
            model: (optional) The prediction model. Currently support pytorch models that returns mean and log variance.
            config: dictionary containing the config parameters for the model.
            device: device used for pytorch models ignored otherwise.
            verbose: if True, print statements with the progress are enabled.
        """

        super(HeteroscedasticRegression).__init__()
        self.config = config
        self.device = device
        self.verbose = verbose
        if model_type == "mlp":
            self.model_type = model_type
            self.model = _MLPNet(
                num_features=self.config["num_features"],
                num_outputs=self.config["num_outputs"],
                num_hidden=self.config["num_hidden"],
            )

        elif model_type == "custom":
            self.model_type = model_type
            self.model = model

        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        return {"model_type": self.model_type, "config": self.config, "model": self.model,
                "device": self.device, "verbose": self.verbose}

    def _loss(self, y_true, y_pred_mu, y_pred_log_var):
        return torch.mean(0.5 * torch.exp(-y_pred_log_var) * torch.abs(y_true - y_pred_mu) ** 2 +
                          0.5 * y_pred_log_var)

    def fit(self, X, y):
        """ Fit the Heteroscedastic Regression model.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        dataset_loader = DataLoader(
            TensorDataset(X,y),
            batch_size=self.config["batch_size"]
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        for epoch in range(self.config["num_epochs"]):
            avg_loss = 0.0
            for batch_x, batch_y in dataset_loader:
                self.model.train()
                batch_y_pred_mu, batch_y_pred_log_var = self.model(batch_x)
                loss = self.model.loss(batch_y, batch_y_pred_mu, batch_y_pred_log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()/len(dataset_loader)

            if self.verbose:
                print("Epoch: {}, loss = {}".format(epoch, avg_loss))

        return self

    def predict(self, X, return_dists=False):
        """
        Obtain predictions for the test points.

        In addition to the mean and lower/upper bounds, also returns epistemic uncertainty (return_epistemic=True)
        and full predictive distribution (return_dists=True).

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            return_dists: If True, the predictive distribution for each instance using scipy distributions is returned.

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
            dists: list of predictive distribution as `scipy.stats` objects with length n_samples.
                Only returned when `return_dists` is True.
        """

        self.model.eval()

        X = torch.from_numpy(X).float().to(self.device)
        dataset_loader = DataLoader(
            X,
            batch_size=self.config["batch_size"]
        )

        y_mean_list = []
        y_log_var_list = []
        for batch_x in dataset_loader:
            batch_y_pred_mu, batch_y_pred_log_var = self.model(batch_x)
            y_mean_list.append(batch_y_pred_mu.data.cpu().numpy())
            y_log_var_list.append(batch_y_pred_log_var.data.cpu().numpy())

        y_mean = np.concatenate(y_mean_list)
        y_log_var = np.concatenate(y_log_var_list)
        y_std = np.sqrt(np.exp(y_log_var))
        y_lower = y_mean - 2.0*y_std
        y_upper = y_mean + 2.0*y_std

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        if return_dists:
            dists = [norm(loc=y_mean[i], scale=y_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_dists',))
            res = Result(*res, y_dists=dists)

        return res
