from collections import namedtuple

import botorch
import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize
from gpytorch.constraints import GreaterThan
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from uq360.algorithms.builtinuq import BuiltinUQ

np.random.seed(42)
torch.manual_seed(42)


class HomoscedasticGPRegression(BuiltinUQ):
    """ A wrapper around Botorch SingleTask Gaussian Process Regression [1]_ with homoscedastic noise.

    References:
        .. [1] https://botorch.org/api/models.html#singletaskgp

    """

    def __init__(self,
                 kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                 likelihood=None,
                 config=None):
        """
        Args:
            kernel: gpytorch kernel function with default set to `RBFKernel` with output scale.
            likelihood: gpytorch likelihood function with default set to `GaussianLikelihood`.
            config: dictionary containing the config parameters for the model.
        """

        super(HomoscedasticGPRegression).__init__()
        self.config = config
        self.kernel = kernel
        self.likelihood = likelihood
        self.model = None
        self.scaler = StandardScaler()
        self.X_bounds = None

    def get_params(self, deep=True):
        return {"kernel": self.kernel, "likelihood": self.likelihood, "config": self.config}

    def fit(self, X, y, **kwargs):
        """
        Fit the GP Regression model.

        Additional arguments relevant for SingleTaskGP fitting can be passed to this function.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values
            **kwargs: Additional arguments relevant for SingleTaskGP fitting.

        Returns:
            self

        """
        y = self.scaler.fit_transform(y)
        X, y = torch.tensor(X), torch.tensor(y)
        self.X_bounds = X_bounds = torch.stack([X.min() * torch.ones(X.shape[1]),
                                X.max() * torch.ones(X.shape[1])])

        X = normalize(X, X_bounds)

        model_homo = SingleTaskGP(train_X=X, train_Y=y, covar_module=self.kernel, likelihood=self.likelihood, **kwargs)
        model_homo.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        model_homo_marginal_log_lik = gpytorch.mlls.ExactMarginalLogLikelihood(model_homo.likelihood, model_homo)
        botorch.fit.fit_gpytorch_model(model_homo_marginal_log_lik)

        model_homo_marginal_log_lik.eval()

        self.model = model_homo_marginal_log_lik
        self.inferred_observation_noise = self.scaler.inverse_transform(self.model.likelihood.noise.detach().numpy()[0].reshape(1,1)).squeeze()

        return self

    def predict(self, X, return_dists=False, return_epistemic=False, return_epistemic_dists=False):
        """
        Obtain predictions for the test points.

        In addition to the mean and lower/upper bounds, also returns epistemic uncertainty (return_epistemic=True)
        and full predictive distribution (return_dists=True).

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            return_dists: If True, the predictive distribution for each instance using scipy distributions is returned.
            return_epistemic: if True, the epistemic upper and lower bounds are returned.
            return_epistemic_dists: If True, the epistemic distribution for each instance using scipy distributions
                is returned.

        Returns:
            namedtuple: A namedtuple that holds

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
        X = torch.tensor(X)

        X_test_norm = normalize(X, self.X_bounds)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.model.posterior(X_test_norm)
            y_mean = posterior.mean
            #y_epi_std = torch.sqrt(posterior.variance)
            y_lower_epistemic, y_upper_epistemic = posterior.mvn.confidence_region()

            predictive_posterior = self.model.model.posterior(X_test_norm, observation_noise=True)
            #y_std = torch.sqrt(predictive_posterior.variance)
            y_lower_total, y_upper_total = predictive_posterior.mvn.confidence_region()

        y_mean, y_lower, y_upper, y_lower_epistemic, y_upper_epistemic = self.scaler.inverse_transform(y_mean.numpy()).squeeze(), \
               self.scaler.inverse_transform(y_lower_total.numpy()).squeeze(),\
               self.scaler.inverse_transform(y_upper_total.numpy()).squeeze(),\
               self.scaler.inverse_transform(y_lower_epistemic.numpy()).squeeze(),\
               self.scaler.inverse_transform(y_upper_epistemic.numpy()).squeeze()

        y_epi_std = (y_upper_epistemic - y_lower_epistemic) / 4.0
        y_std = (y_upper_total - y_lower_total) / 4.0

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        if return_epistemic:
            Result = namedtuple('res', Result._fields + ('y_lower_epistemic', 'y_upper_epistemic',))
            res = Result(*res, y_lower_epistemic=y_lower_epistemic, y_upper_epistemic=y_upper_epistemic)

        if return_dists:
            dists = [norm(loc=y_mean[i], scale=y_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_dists',))
            res = Result(*res, y_dists=dists)

        if return_epistemic_dists:
            epi_dists = [norm(loc=y_mean[i], scale=y_epi_std[i]) for i in range(y_mean.shape[0])]
            Result = namedtuple('res', Result._fields + ('y_epistemic_dists',))
            res = Result(*res, y_epistemic_dists=epi_dists)

        return res

