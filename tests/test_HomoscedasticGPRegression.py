import unittest

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


class TestHomoscedasticGPRegression(unittest.TestCase):

    def _generate_mock_data(self, n_samples, n_features):
        from sklearn.datasets import make_regression
        return make_regression(n_samples, n_features, random_state=42)

    def test_fit_predict_and_metrics(self):

        from uq360.algorithms.homoscedastic_gaussian_process_regression import HomoscedasticGPRegression
        from uq360.metrics import compute_regression_metrics
        X, y = self._generate_mock_data(200, 3)
        y = y.reshape(-1, 1)

        uq_model = HomoscedasticGPRegression()
        uq_model.fit(X, y)
        yhat, yhat_lb, yhat_ub, yhat_lb_epi, yhat_ub_epi, yhat_dists = uq_model.predict(X, return_dists=True, return_epistemic=True)

        results = compute_regression_metrics(y.ravel(), yhat, yhat_lb, yhat_ub)

        coverage = results["picp"]
        avg_width = results["mpiw"]
        rmse = results["rmse"]
        nll = results["nll"]
        auucc_gain = results["auucc_gain"]

        assert (coverage > 0.0)


if __name__ == '__main__':
    unittest.main()
