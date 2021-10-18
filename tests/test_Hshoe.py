import unittest

import numpy as np
import torch

from uq360.algorithms.variational_bayesian_neural_networks.bnn import BnnRegression, BnnClassification
from uq360.models.bayesian_neural_networks.layer_utils import InvGammaHalfCauchyLayer
from uq360.models.bayesian_neural_networks.layers import HorseshoeLayer


class TestHC(unittest.TestCase):

    def test_hcauchy_init(self):
        a = InvGammaHalfCauchyLayer(out_features=10, b=1)
        assert a.bhat.shape[0] == 10
        assert a.mu.shape == a.bhat.shape
        assert a.log_sigma.shape == a.bhat.shape
        b = InvGammaHalfCauchyLayer(out_features=1, b=10)
        assert b.bhat.shape[0] == 1
        c = InvGammaHalfCauchyLayer(out_features=1000, b=1e-5)
        assert c.bhat.shape[0] == 1000

    def test_hshoe_layer(self):
        out_features = 2
        hs = HorseshoeLayer(in_features=10, out_features=out_features, scale=1e-5)
        assert(hs.nodescales.mu.shape[0] == out_features)
        assert(hs.layerscale.mu.shape[0] == 1)
        x = torch.FloatTensor(5, 10).normal_()
        y = hs.forward(x)
        assert y.shape[0] == 5
        assert y.shape[1] == out_features

    def test_uncertainty_classification(self):
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from uq360.metrics.classification_metrics import entropy_based_uncertainty_decomposition, \
            compute_classification_metrics
        n_samples = 200
        n_features = 5
        n_classes = 4
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        config = {'ip_dim': n_features, 'op_dim': n_classes, 'num_nodes': 5, 'num_layers': 1, 'num_epochs': 10,
                  'step_size': 1e-2}
        uq_bnn = BnnClassification(config=config)
        uq_bnn.fit(X_train, y_train)
        y_pred, y_prob, y_prob_var, y_prob_samples = uq_bnn.predict(X=X_test, mc_samples=10)
        res = compute_classification_metrics(y_true=y_test, y_prob=y_prob, option='all')
        self.assertCountEqual(res.keys(), ['ece', 'aurrrc', 'auroc', 'nll', 'brier', 'accuracy'])

        res = compute_classification_metrics(y_true=y_test, y_prob=y_prob, option='ece')
        assert "ece" in res

        res = compute_classification_metrics(y_true=y_test, y_prob=y_prob,
                                       option=['aurrrc', 'auroc', 'nll', 'brier', 'accuracy'])
        self.assertCountEqual(res.keys(), ['aurrrc', 'auroc', 'nll', 'brier', 'accuracy'])

        total_uq, aleatoric_uq, epistemic_uq = entropy_based_uncertainty_decomposition(y_prob_samples)
        assert total_uq.shape[0] == X_test.shape[0]
        assert len(total_uq.shape) == 1
        assert len(total_uq.shape) == len(aleatoric_uq.shape) == len(epistemic_uq.shape)

    def test_uncertainty_regression(self):
        from uq360.utils.generate_1D_regression_data import make_data_sine
        from uq360.metrics.regression_metrics import compute_regression_metrics

        viz = False
        x_train, y_train, x_val, y_val, train_stats = make_data_sine(0, data_count=500) # 20% for training ; 80 % for testing

        config = {'ip_dim': 1, 'op_dim': 1, 'num_nodes': 50, 'num_layers': 1, 'num_epochs': 100,
                  'step_size': 1e-2}  # network params, learning epochs, and learning rate
        uq_bnn = BnnRegression(config=config)
        uq_bnn.fit(x_train, y_train)
        y, y_lb, y_ub, y_epi_lb, y_epi_ub = uq_bnn.predict(x_val)

        res = compute_regression_metrics(y_true=y_val.numpy(), y_mean=y, y_lower=y_lb, y_upper=y_ub)
        self.assertCountEqual(res.keys(), ["rmse", "nll", "auucc_gain", "picp", "mpiw", "r2"])
        res = compute_regression_metrics(y_true=y_val.numpy(), y_mean=y, y_lower=y_lb, y_upper=y_ub, option='picp')
        assert 'picp' in res
        res = compute_regression_metrics(y_true=y_val.numpy(), y_mean=y, y_lower=y_lb, y_upper=y_ub, option=['mpiw', 'rmse'])
        self.assertCountEqual(res.keys(), ['mpiw', 'rmse'])

        if viz:
            idx = np.argsort(x_val.numpy().ravel())
            import matplotlib.pyplot as plt
            plt.title("BNN")
            plt.plot(x_val[idx], y[idx], 'ro-')
            plt.fill_between(x_val.numpy().ravel()[idx], y_lb[idx], y_ub[idx],
                             alpha=0.2, label='Total', color='k')
            plt.fill_between(x_val.numpy().ravel()[idx], y_epi_lb[idx], y_epi_ub[idx],
                             alpha=0.2, label='Epistemic', color='b')

        # horseshoe
        config['hshoe_scale'] = 1e-1
        uq_hs_bnn = BnnRegression(config=config, prior="Hshoe")
        uq_hs_bnn.fit(x_train, y_train)
        y, y_lb, y_ub, y_epi_lb, y_epi_ub = uq_hs_bnn.predict(x_val)
        if viz:
            plt.figure()
            plt.title("Horseshoe BNN")
            plt.plot(x_val[idx], y[idx], 'ro-')
            plt.fill_between(x_val.numpy().ravel()[idx], y_lb[idx], y_ub[idx],
                             alpha=0.2, label='Total', color='k')
            plt.fill_between(x_val.numpy().ravel()[idx], y_epi_lb[idx], y_epi_ub[idx],
                             alpha=0.2, label='Epistemic', color='b')

        # regularized horseshoe
        uq_reghs_bnn = BnnRegression(config=config, prior="RegHshoe")
        uq_reghs_bnn.fit(x_train, y_train)
        y, y_lb, y_ub, y_epi_lb, y_epi_ub = uq_reghs_bnn.predict(x_val)
        if viz:
            plt.figure()
            plt.title("Regularized Horseshoe BNN")
            plt.plot(x_val[idx], y[idx], 'ro-')
            plt.fill_between(x_val.numpy().ravel()[idx], y_lb[idx], y_ub[idx],
                             alpha=0.2, label='Total', color='k')
            plt.fill_between(x_val.numpy().ravel()[idx], y_epi_lb[idx], y_epi_ub[idx],
                             alpha=0.2, label='Epistemic', color='b')
            plt.show()


if __name__ == '__main__':
    unittest.main()