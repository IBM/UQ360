from collections import namedtuple

import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from uq360.algorithms.heteroscedastic_regression import HeteroscedasticRegression
from uq360.algorithms.builtinuq import BuiltinUQ

class _Ensemble(torch.nn.Module): 
    """_Ensemble creates an ensemble of models and returns the mean and the log variance of the ensemble
    (following convention of heteroscedastic regression)"""
    #TODO generalize to ensemble of any models using this hidden class

    def __init__(self, num_models, model_function, *args, **kwargs):
        super(_Ensemble, self).__init__()
        self.models_ = []
        for _ in range(num_models):
            self.models_.append(model_function(*args, **kwargs))
    
    def forward(self, x):
        mu_list = []
        var_list = []

        for regression in self.models_:
            mu, log_var = regression.model(x)
            mu_list.append(mu)
            var_list.append(torch.exp(log_var))

        mu_stack = torch.stack(mu_list)
        var_stack = torch.stack(var_list)
        
        num_models = len(mu_stack)

        mu_ensemble = sum(mu_stack) / num_models
        var_ensemble = sum(var_stack + mu_stack**2) / num_models - mu_ensemble**2

        return mu_ensemble, torch.log(var_ensemble)

class EnsembleHeteroscedasticRegression(BuiltinUQ):
    """Ensemble Regression assumes an ensemble of models of Gaussian form for the predictive distribution and 
    returns the mean and log variance of the ensemble of Gaussians.
    """

    def __init__(self, model_type=None, config=None, device=None, verbose=True):
        """ Initializer for Ensemble of heteroscedastic regression.
        Args:
            model_type: The base model used for predicting a quantile. Currently supported values are [heteroscedasticregression].
            config: dictionary containing the config parameters for the model.
            device: device used for pytorch models ignored otherwise.
        """

        super(EnsembleHeteroscedasticRegression).__init__()
        self.config = config
        self.device = device
        self.verbose = verbose        
        if model_type == "ensembleheteroscedasticregression":
            self.model_type = model_type
            self.model = _Ensemble(
                num_models=self.config["num_models"],
                model_function=HeteroscedasticRegression,
                **self.config["model_kwargs"]
            )

        else:
            raise NotImplementedError
    
    def fit(self, X, y):
        """ Fit the Ensemble of Heteroscedastic Regression models.
        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values
        Returns:
            self
        """
        #TODO:paralellize model training for the ensemble

        for i in range(len(self.model.models_)):
            torch.manual_seed(i)
            torch.cuda.manual_seed(i)
            torch.backends.cudnn.enabled=False
            torch.backends.cudnn.deterministic=True

            self.verbose and print(f"\nTraining model {i}\n")
            self.model.models_[i] = self.model.models_[i].fit(X, y)
        
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
            batch_size=self.config["model_kwargs"]["config"]["batch_size"]
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
            