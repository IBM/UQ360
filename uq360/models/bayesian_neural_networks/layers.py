"""
  Contains implementations of various Bayesian layers
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from uq360.models.bayesian_neural_networks.layer_utils import InvGammaHalfCauchyLayer, InvGammaLayer

td = torch.distributions


def reparam(mu, logvar, do_sample=True, mc_samples=1):
    if do_sample:
        std = torch.exp(0.5 * logvar)
        eps = torch.FloatTensor(std.size()).normal_()
        sample = mu + eps * std
        for _ in np.arange(1, mc_samples):
            sample += mu + eps * std
        return sample / mc_samples
    else:
        return mu


class BayesianLinearLayer(torch.nn.Module):
    """
    Affine layer with N(0, v/H) or N(0, user specified v) priors on weights and
    fully factorized variational Gaussian approximation
    """

    def __init__(self, in_features, out_features, cuda=False, init_weight=None, init_bias=None, prior_stdv=None):
        super(BayesianLinearLayer, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features

        # weight mean params
        self.weights = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        # weight variance params
        self.weights_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))

        # numerical stability
        self.fudge_factor = 1e-8
        if not prior_stdv:
            # We will use a N(0, 1/num_inputs) prior over weights
            self.prior_stdv = torch.FloatTensor([1. / np.sqrt(self.weights.size(1))])
        else:
            self.prior_stdv = torch.FloatTensor([prior_stdv])
        # self.prior_stdv = torch.Tensor([1. / np.sqrt(1e+3)])
        self.prior_mean = torch.FloatTensor([0.])
        # for Bias use a prior of N(0, 1)
        self.prior_bias_stdv = torch.FloatTensor([1.])
        self.prior_bias_mean = torch.FloatTensor([0.])

        # init params either random or with pretrained net
        self.init_parameters(init_weight, init_bias)

    def init_parameters(self, init_weight, init_bias):
        # init means
        if init_weight is not None:
            self.weights.data = torch.Tensor(init_weight)
        else:
            self.weights.data.normal_(0, np.float(self.prior_stdv.numpy()[0]))

        if init_bias is not None:
            self.bias.data = torch.Tensor(init_bias)
        else:
            self.bias.data.normal_(0, 1)

        # init variances
        self.weights_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def forward(self, x, do_sample=True, scale_variances=False):
        # local reparameterization trick
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        if scale_variances:
            activ = reparam(mu_activations, var_activations.log() - np.log(self.in_features), do_sample=do_sample)
        else:
            activ = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return activ

    def kl(self):
        """
        KL divergence (q(W) || p(W))
        :return:
        """
        weights_logvar = self.weights_logvar
        kld_weights = self.prior_stdv.log() - weights_logvar.mul(0.5) + \
                      (weights_logvar.exp() + (self.weights.pow(2) - self.prior_mean)) / (
                                  2 * self.prior_stdv.pow(2)) - 0.5
        kld_bias = self.prior_bias_stdv.log() - self.bias_logvar.mul(0.5) + \
                   (self.bias_logvar.exp() + (self.bias.pow(2) - self.prior_bias_mean)) / (
                               2 * self.prior_bias_stdv.pow(2)) \
                   - 0.5
        return kld_weights.sum() + kld_bias.sum()


class HorseshoeLayer(BayesianLinearLayer):
    """
    Uses non-centered parametrization. w_k = v*tau_k*beta_k where k indexes an output unit and w_k and beta_k
    are vectors of all weights incident into the unit
    """
    def __init__(self, in_features, out_features, cuda=False, scale=1.):
        super(HorseshoeLayer, self).__init__(in_features, out_features)
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.nodescales = InvGammaHalfCauchyLayer(out_features=out_features, b=1.)
        self.layerscale = InvGammaHalfCauchyLayer(out_features=1, b=scale)
        # prior on beta is N(0, I) when employing non centered parameterization
        self.prior_stdv = torch.Tensor([1])
        self.prior_mean = torch.Tensor([0.])

    def forward(self,  x, do_sample=True, debug=False, eps_scale=None, eps_w=None):
        # At a particular unit k, preactivation_sample = scale_sample * pre_activation_sample
        # sample scales
        scale_mean = 0.5 * (self.nodescales.mu + self.layerscale.mu)
        scale_var = 0.25 * (self.nodescales.log_sigma.exp() ** 2 + self.layerscale.log_sigma.exp() ** 2)
        scale_sample = reparam(scale_mean, scale_var.log(), do_sample=do_sample).exp()
        # sample preactivations
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ_sample = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return scale_sample * activ_sample

    def kl(self):
        return super(HorseshoeLayer, self).kl() + self.nodescales.kl() + self.layerscale.kl()

    def fixed_point_updates(self):
        self.nodescales.fixed_point_updates()
        self.layerscale.fixed_point_updates()


class RegularizedHorseshoeLayer(HorseshoeLayer):
    """
    Uses the regularized Horseshoe distribution. The regularized Horseshoe soft thresholds the tails of the Horseshoe.
    For all weights w_k incident upon node k in the layer we have:
    w_k ~ N(0, (tau_k * v)^2 I) N(0, c^2 I), c^2 ~ InverseGamma(c_a, b).
    c^2 controls the scale of the thresholding. As c^2 -> infinity, the regularized Horseshoe -> Horseshoe.
    """

    def __init__(self, in_features, out_features, cuda=False, scale=1., c_a=2., c_b=6.):
        super(RegularizedHorseshoeLayer, self).__init__(in_features, out_features, cuda=cuda, scale=scale)
        self.c = InvGammaLayer(a=c_a, b=c_b)

    def forward(self, x, do_sample=True, **kwargs):
        # At a particular unit k, preactivation_sample = scale_sample * pre_activation_sample
        # sample regularized scales
        scale_mean = self.nodescales.mu + self.layerscale.mu
        scale_var = self.nodescales.log_sigma.exp() ** 2 + self.layerscale.log_sigma.exp() ** 2
        scale_sample = reparam(scale_mean, scale_var.log(), do_sample=do_sample).exp()
        c_sample = reparam(self.c.mu, 2 * self.c.log_sigma, do_sample=do_sample).exp()
        regularized_scale_sample = (c_sample * scale_sample) / (c_sample + scale_sample)
        # sample preactivations
        mu_activations = F.linear(x, self.weights, self.bias)
        var_activations = F.linear(x.pow(2), self.weights_logvar.exp(), self.bias_logvar.exp())
        activ_sample = reparam(mu_activations, var_activations.log(), do_sample=do_sample)
        return torch.sqrt(regularized_scale_sample) * activ_sample

    def kl(self):
        return super(RegularizedHorseshoeLayer, self).kl() + self.c.kl()


class NodeSpecificRegularizedHorseshoeLayer(RegularizedHorseshoeLayer):
    """
    Uses the regularized Horseshoe distribution. The regularized Horseshoe soft thresholds the tails of the Horseshoe.
    For all weights w_k incident upon node k in the layer we have:
    w_k ~ N(0, (tau_k * v)^2 I) N(0, c_k^2 I), c_k^2 ~ InverseGamma(a, b).
    c_k^2 controls the scale of the thresholding. As c_k^2 -> infinity, the regularized Horseshoe -> Horseshoe
    Note that we now have a per-node c_k.
    """

    def __init__(self, in_features, out_features, cuda=False, scale=1., c_a=2., c_b=6.):
        super(NodeSpecificRegularizedHorseshoeLayer, self).__init__(in_features, out_features, cuda=cuda, scale=scale)
        self.c = InvGammaLayer(a=c_a, b=c_b, out_features=out_features)




