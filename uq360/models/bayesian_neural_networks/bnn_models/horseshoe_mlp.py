from abc import ABC

import numpy as np
import torch
from torch import nn

from uq360.models.bayesian_neural_networks.layers import HorseshoeLayer, BayesianLinearLayer, RegularizedHorseshoeLayer
from uq360.models.noise_models.homoscedastic_noise_models import GaussianNoiseGammaPrecision
import numpy as np
td = torch.distributions


class HshoeBNN(nn.Module, ABC):
    """
     Bayesian neural network with Horseshoe layers.
    """
    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu', num_layers=1,
                 hshoe_scale=1e-1, use_reg_hshoe=False):
        if use_reg_hshoe:
            layer = RegularizedHorseshoeLayer
        else:
            layer = HorseshoeLayer
        super(HshoeBNN, self).__init__()
        self.num_layers = num_layers
        if activation_type == 'relu':
            # activation
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            print("Activation Type not supported")
        self.fc_hidden = []
        self.fc1 = layer(ip_dim, num_nodes, scale=hshoe_scale)
        for _ in np.arange(self.num_layers - 1):
            self.fc_hidden.append(layer(num_nodes, num_nodes))
        self.fc_out = BayesianLinearLayer(num_nodes, op_dim)
        self.noise_layer = None

    def forward(self, x, do_sample=True):
        x = self.fc1(x, do_sample=do_sample)
        x = self.activation(x)
        for layer in self.fc_hidden:
            x = layer(x, do_sample=do_sample)
            x = self.activation(x)
        return self.fc_out(x, do_sample=do_sample, scale_variances=True)

    def kl_divergence_w(self):
        kld = self.fc1.kl() + self.fc_out.kl()
        for layer in self.fc_hidden:
            kld += layer.kl()
        return kld

    def fixed_point_updates(self):
        if hasattr(self.fc1, 'fixed_point_updates'):
            self.fc1.fixed_point_updates()
        if hasattr(self.fc_out, 'fixed_point_updates'):
            self.fc_out.fixed_point_updates()
        for layer in self.fc_hidden:
            if hasattr(layer, 'fixed_point_updates'):
                layer.fixed_point_updates()

    def prior_predictive_samples(self, n_sample=100):
        n_eval = 1000
        x = torch.linspace(-2, 2, n_eval)[:, np.newaxis]
        y = np.zeros([n_sample, n_eval])
        for i in np.arange(n_sample):
            y[i] = self.forward(x).data.numpy().ravel()
        return x.data.numpy(), y

    ### get and set weights ###
    def get_weights(self):
        assert len(self.fc_hidden) == 0 # only works for one layer networks.
        weight_dict = {}
        weight_dict['layerip_means'] = torch.cat([self.fc1.weights, self.fc1.bias.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerip_logvar'] = torch.cat([self.fc1.weights_logvar, self.fc1.bias_logvar.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerop_means'] = torch.cat([self.fc_out.weights, self.fc_out.bias.unsqueeze(1)], dim=1).data.numpy()
        weight_dict['layerop_logvar'] = torch.cat([self.fc_out.weights_logvar, self.fc_out.bias_logvar.unsqueeze(1)], dim=1).data.numpy()
        return weight_dict

    def set_weights(self, weight_dict):
        assert len(self.fc_hidden) == 0  # only works for one layer networks.
        to_param = lambda x: nn.Parameter(torch.Tensor(x))
        self.fc1.weights = to_param(weight_dict['layerip_means'][:, :-1])
        self.fc1.weights = to_param(weight_dict['layerip_logvar'][:, :-1])
        self.fc1.bias = to_param(weight_dict['layerip_means'][:, -1])
        self.fc1.bias_logvar = to_param(weight_dict['layerip_logvar'][:, -1])

        self.fc_out.weights = to_param(weight_dict['layerop_means'][:, :-1])
        self.fc_out.weights = to_param(weight_dict['layerop_logvar'][:, :-1])
        self.fc_out.bias = to_param(weight_dict['layerop_means'][:, -1])
        self.fc_out.bias_logvar = to_param(weight_dict['layerop_logvar'][:, -1])


class HshoeRegressionNet(HshoeBNN, ABC):
    """
    Horseshoe net with N(y_true | f(x, w), \lambda^-1); \lambda ~ Gamma(a, b) likelihoods.
    """
    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                     num_layers=1, hshoe_scale=1e-5, use_reg_hshoe=False):
        super(HshoeRegressionNet, self).__init__(ip_dim=ip_dim, op_dim=op_dim,
                                                    num_nodes=num_nodes, activation_type=activation_type,
                                                    num_layers=num_layers,
                                                    hshoe_scale=hshoe_scale,
                                                    use_reg_hshoe=use_reg_hshoe)
        self.noise_layer = GaussianNoiseGammaPrecision(a0=6., b0=6.)

    def likelihood(self, x=None, y=None):
        out = self.forward(x)
        return -self.noise_layer.loss(y_pred=out, y_true=y)

    def neg_elbo(self, num_batches, x=None, y=None):
        # scale the KL terms by number of batches so that the minibatch elbo is an unbiased estiamte of the true elbo.
        Elik = self.likelihood(x, y)
        neg_elbo = (self.kl_divergence_w() + self.noise_layer.kl()) / num_batches - Elik
        return neg_elbo

    def mse(self, x, y):
        """
        scaled rmse (scaled by 1 / std_y**2)
        """
        E_noise_precision = 1. / self.noise_layer.get_noise_var()
        return (0.5 * E_noise_precision * (self.forward(x, do_sample=False) - y)**2).sum()

    def get_noise_var(self):
        return self.noise_layer.get_noise_var()


class HshoeClassificationNet(HshoeBNN, ABC):
    """
    Horseshoe net with Categorical(y_true | f(x, w)) likelihoods. Use for classification.
    """
    def __init__(self, ip_dim=1, op_dim=1, num_nodes=50, activation_type='relu',
                 num_layers=1, hshoe_scale=1e-5, use_reg_hshoe=False):
        super(HshoeClassificationNet, self).__init__(ip_dim=ip_dim, op_dim=op_dim,
                                                    num_nodes=num_nodes, activation_type=activation_type,
                                                    num_layers=num_layers,
                                                     hshoe_scale=hshoe_scale,
                                                     use_reg_hshoe=use_reg_hshoe)
        self.noise_layer = torch.nn.CrossEntropyLoss(reduction='sum')

    def likelihood(self, x=None, y=None):
        out = self.forward(x)
        return -self.noise_layer(out, y)

    def neg_elbo(self, num_batches, x=None, y=None):
        # scale the KL terms by number of batches so that the minibatch elbo is an unbiased estiamte of the true elbo.
        Elik = self.likelihood(x, y)
        neg_elbo = (self.kl_divergence_w()) / num_batches - Elik
        return neg_elbo




