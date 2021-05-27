import unittest

import numpy as np
import torch

from uq360.algorithms.variational_bayesian_neural_networks.bnn import BnnRegression
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

    def test_uncertainty(self):
        from uq360.utils.generate_1D_regression_data import make_data_sine
        viz = False
        x_train, y_train, x_val, y_val, train_stats = make_data_sine(0, data_count=500) # 20% for training ; 80 % for testing

        config = {'ip_dim': 1, 'op_dim': 1, 'num_nodes': 50, 'num_layers': 1, 'num_epochs': 1000,
                  'step_size': 1e-2}  # network params, learning epochs, and learning rate
        uq_bnn = BnnRegression(config=config)
        uq_bnn.fit(x_train, y_train)
        y, y_lb, y_ub, y_epi_lb, y_epi_ub = uq_bnn.predict(x_val)

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