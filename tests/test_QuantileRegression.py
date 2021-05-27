import unittest

import numpy as np

np.random.seed(42)


class TestQuantileRegression(unittest.TestCase):

    def _generate_mock_data(self, n_samples, n_features):
        from sklearn.datasets import make_regression
        return make_regression(n_samples, n_features, random_state=42)

    def test_fit_predict_and_metrics(self):
        from uq360.algorithms.quantile_regression import QuantileRegression
        from uq360.metrics import picp, mpiw
        X, y = self._generate_mock_data(200, 3)
        config = {'loss': 'quantile', 'alpha': 0.95, 'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.001,
                  'min_samples_leaf': 10, 'min_samples_split': 10, 'random_state': 42}
        uq_model = QuantileRegression(model_type='gbr', config=config)
        uq_model.fit(X, y)
        yhat, yhat_lb, yhat_ub = uq_model.predict(X)

        coverage = picp(y, yhat_lb, yhat_ub)
        avg_width = mpiw(yhat_lb, yhat_ub)

        assert((coverage > 0.0) and (avg_width > 0.0))


if __name__ == '__main__':
    unittest.main()
