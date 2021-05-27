from collections import namedtuple

import numpy as np

from uq360.algorithms.posthocuq import PostHocUQ


class InfinitesimalJackknife(PostHocUQ):
    """
    Performs a first order Taylor series expansion around MLE / MAP fit.
    Requires the model being probed to be twice differentiable.
    """
    def __init__(self, params, gradients, hessian, config):
        """ Initialize IJ.
        Args:
            params: MLE / MAP fit around which uncertainty is sought. d*1
            gradients: Per data point gradients, estimated at the MLE / MAP fit. d*n
            hessian: Hessian evaluated at the MLE / MAP fit. d*d
        """

        super(InfinitesimalJackknife).__init__()
        self.params_one = params
        self.gradients = gradients
        self.hessian = hessian
        self.d, self.n = gradients.shape
        self.dParams_dWeights = -np.linalg.solve(self.hessian, self.gradients)
        self.approx_dParams_dWeights = -np.linalg.solve(np.diag(np.diag(self.hessian)), self.gradients)
        self.w_one = np.ones([self.n])
        self.config = config

    def get_params(self, deep=True):
        return {"params": self.params, "config": self.config, "gradients": self.gradients,
                "hessian": self.hessian}

    def _process_pretrained_model(self, *argv, **kwargs):
        pass

    def get_parameter_uncertainty(self):
        if (self.config['resampling_strategy'] == "jackknife") or (self.config['resampling_strategy'] == "jackknife+"):
            w_query = np.ones_like(self.w_one)
            resampled_params = np.zeros([self.n, self.d])
            for i in np.arange(self.n):
                w_query[i] = 0
                resampled_params[i] = self.ij(w_query)
                w_query[i] = 1
            return np.cov(resampled_params), resampled_params
        elif self.config['resampling_strategy'] == "bootstrap":
            pass
        else:
            raise NotImplementedError("Only jackknife, jackknife+, and bootstrap resampling strategies are supported")

    def predict(self, X, model):
        """
        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            model: model object, must implement a set_parameters function

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.

        """
        n, _ = X.shape
        y_all = model.predict(X)
        _, d_out = y_all.shape
        params_cov, params = self.get_parameter_uncertainty()
        if d_out > 1:
            print("Quantiles are computed independently for each dimension. May not be accurate.")
        y = np.zeros([params.shape[0], n, d_out])
        for i in np.arange(params.shape[0]):
            model.set_parameters(params[i])
            y[i] = model.predict(X)
        y_lower = np.quantile(y, q=0.5 * self.config['alpha'], axis=0)
        y_upper = np.quantile(y, q=(1. - 0.5 * self.config['alpha']), axis=0)
        y_mean = y.mean(axis=0)

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        return res

    def ij(self, w_query):
        """
        Args:
            w_query: A n*1 vector to query parameters at.
        Return:
            new parameters at w_query
        """
        assert w_query.shape[0] == self.n
        return self.params_one + self.dParams_dWeights @ (w_query-self.w_one).T

    def approx_ij(self, w_query):
        """
        Args:
            w_query: A n*1 vector to query parameters at.
        Return:
            new parameters at w_query
        """
        assert w_query.shape[0] == self.n
        return self.params_one + self.approx_dParams_dWeights @ (w_query-self.w_one).T