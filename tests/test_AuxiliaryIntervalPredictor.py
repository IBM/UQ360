import unittest

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


class TestAuxiliaryIntervalPredictor(unittest.TestCase):

    def _generate_mock_data(self, n_samples, n_features):
        from sklearn.datasets import make_regression
        return make_regression(n_samples, n_features, random_state=42)

    def test_fit_predict_and_metrics(self):

        from uq360.algorithms.auxiliary_interval_predictor import AuxiliaryIntervalPredictor
        from uq360.metrics import compute_regression_metrics
        X, y = self._generate_mock_data(200, 3)
        y = y.reshape(-1, 1)
        config = {"num_features": X.shape[1], "num_outputs": y.shape[1], "num_hidden": 50, "batch_size": 32, "num_epochs": 2,
                  "lr": 0.001, "num_main_iters": 2, "num_aux_iters": 2, "num_outer_iters": 2,
                  "lambda_noise": 1.0, "lambda_sharpness": 1.0, "lambda_match": 1.0,
                  "calibration_alpha": 0.90}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        uq_model = AuxiliaryIntervalPredictor(model_type='mlp', config=config, device=device)
        uq_model.fit(X, y)
        yhat, yhat_lb, yhat_ub, yhat_dists = uq_model.predict(X, return_dists=True)

        results = compute_regression_metrics(y, yhat, yhat_lb, yhat_ub)

        coverage = results["picp"]
        avg_width = results["mpiw"]
        rmse = results["rmse"]
        nll = results["nll"]
        auucc_gain = results["auucc_gain"]

        assert (coverage > 0.0)


if __name__ == '__main__':
    unittest.main()
