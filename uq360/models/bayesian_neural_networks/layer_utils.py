"""
  Contains implementations of various utilities used by Horseshoe Bayesian layers
"""
import numpy as np
import torch
from torch.nn import Parameter

td = torch.distributions
gammaln = torch.lgamma


def diag_gaussian_entropy(log_std, D):
    return 0.5 * D * (1.0 + torch.log(2 * np.pi)) + torch.sum(log_std)


def inv_gamma_entropy(a, b):
    return torch.sum(a + torch.log(b) + torch.lgamma(a) - (1 + a) * torch.digamma(a))


def log_normal_entropy(log_std, mu, D):
    return torch.sum(log_std + mu + 0.5) + (D / 2) * np.log(2 * np.pi)


class InvGammaHalfCauchyLayer(torch.nn.Module):
    """
    Uses the inverse Gamma parameterization of the half-Cauchy distribution.
    a ~ C^+(0, b)  <==> a^2 ~ IGamma(0.5, 1/lambda), lambda ~ IGamma(0.5, 1/b^2), where lambda is an
    auxiliary latent variable.
    Uses a factorized variational approximation q(ln a^2)q(lambda) = N(mu, sigma^2) IGamma(ahat, bhat).
    This layer places a half Cauchy prior on the scales of each output node of the layer.
    """
    def __init__(self, out_features, b):
        """
        :param out_fatures: number of output nodes in the layer.
        :param b: scale of the half Cauchy
        """
        super(InvGammaHalfCauchyLayer, self).__init__()
        self.b = b
        self.out_features = out_features
        # variational parameters for q(ln a^2)
        self.mu = Parameter(torch.FloatTensor(out_features))
        self.log_sigma = Parameter(torch.FloatTensor(out_features))
        # self.log_sigma = torch.FloatTensor(out_features)
        # variational parameters for q(lambda). These will be updated via fixed point updates, hence not parameters.
        self.ahat = torch.FloatTensor([1.])  # The posterior parameter is always 1.
        self.bhat = torch.ones(out_features) * (1.0 / self.b ** 2)
        self.const = torch.FloatTensor([0.5])
        self.initialize_from_prior()

    def initialize_from_prior(self):
        """
        Initializes variational parameters by sampling from the prior.
        """
        # sample from half cauchy and log to initialize the mean of the log normal
        sample = np.abs(self.b * (np.random.randn(self.out_features) / np.random.randn(self.out_features)))
        self.mu.data = torch.FloatTensor(np.log(sample))
        self.log_sigma.data = torch.FloatTensor(np.random.randn(self.out_features) - 10.)

    def expectation_wrt_prior(self):
        """
        Computes E[ln p(a^2 | lambda)] + E[ln p(lambda)]
        """
        expected_a_given_lambda = -gammaln(self.const) - 0.5 * (torch.log(self.bhat) - torch.digamma(self.ahat)) + (
                -0.5 - 1.) * self.mu - torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2) * (self.ahat / self.bhat)
        expected_lambda = -gammaln(self.const) - 2 * 0.5 * np.log(self.b) + (-self.const - 1.) * (
                torch.log(self.bhat) - torch.digamma(self.ahat)) - (1. / self.b ** 2) * (self.ahat / self.bhat)
        return torch.sum(expected_a_given_lambda) + torch.sum(expected_lambda)

    def entropy(self):
        """
        Computes entropy of q(ln a^2) and q(lambda)
        """
        return self.entropy_lambda() + self.entropy_a2()

    def entropy_lambda(self):
        return inv_gamma_entropy(self.ahat, self.bhat)

    def entropy_a2(self):
        return log_normal_entropy(self.log_sigma, self.mu, self.out_features)

    def kl(self):
        """
        Computes KL(q(ln(a^2)q(lambda) || IG(a^2 | 0.5, 1/lambda) IG(lambda | 0.5, 1/b^2))
        """
        return -self.expectation_wrt_prior() - self.entropy()

    def fixed_point_updates(self):
        # update lambda moments
        self.bhat = torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2) + (1. / self.b ** 2)


class InvGammaLayer(torch.nn.Module):
    """
    Approximates the posterior of c^2 with prior IGamma(c^2 | a , b)
    using a log Normal approximation q(ln c^2) = N(mu, sigma^2)
    """

    def __init__(self, a, b, out_features=1):
        super(InvGammaLayer, self).__init__()
        self.a = torch.FloatTensor([a])
        self.b = torch.FloatTensor([b])
        # variational parameters for q(ln c^2)
        self.mu = Parameter(torch.FloatTensor(out_features))
        self.log_sigma = Parameter(torch.FloatTensor(out_features))
        self.out_features = out_features
        self.initialize_from_prior()

    def initialize_from_prior(self):
        """
        Initializes variational parameters by sampling from the prior.
        """
        self.mu.data = torch.log(self.b / (self.a + 1) * torch.ones(self.out_features)) # initialize at the mode
        self.log_sigma.data = torch.FloatTensor(np.random.randn(self.out_features) - 10.)

    def expectation_wrt_prior(self):
        """
        Computes E[ln p(c^2 | a, b)]
        """
        # return self.c_a * np.log(self.c_b) - gammaln(self.c_a) + (
        #                                             - self.c_a - 1) * c_mu - self.c_b * Ecinv
        return self.a * torch.log(self.b) - gammaln(self.a) + (- self.a - 1) \
               * self.mu - self.b * torch.exp(-self.mu + 0.5 * self.log_sigma.exp() ** 2)

    def entropy(self):
        return log_normal_entropy(self.log_sigma, self.mu, 1)

    def kl(self):
        """
        Computes KL(q(ln(c^2) || IG(c^2 | a, b))
        """
        return -self.expectation_wrt_prior().sum() - self.entropy()
