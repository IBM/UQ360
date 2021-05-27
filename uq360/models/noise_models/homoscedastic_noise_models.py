import math

import numpy as np
import torch
from scipy.special import gammaln
from uq360.models.noise_models.noisemodel import AbstractNoiseModel
from torch.nn import Parameter

td = torch.distributions


def transform(a):
    return torch.log(1 + torch.exp(a))


class GaussianNoiseGammaPrecision(torch.nn.Module, AbstractNoiseModel):
    """
        N(y_true | f(x, w), \lambda^-1); \lambda ~ Gamma(a, b).
        Uses a variational approximation; q(lambda) = Gamma(ahat, bhat)
    """

    def __init__(self, a0=6, b0=6, cuda=False):
        super(GaussianNoiseGammaPrecision, self).__init__()
        self.cuda = cuda
        self.a0 = a0
        self.b0 = b0
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))
        # variational parameters
        self.ahat = Parameter(torch.FloatTensor([10.]))
        self.bhat = Parameter(torch.FloatTensor([3.]))

    def loss(self, y_pred=None, y_true=None):
        """
        computes -1 *  E_q(\lambda)[ln N (y_pred | y_true, \lambda^-1)], where q(lambda) = Gamma(ahat, bhat)
        :param y_pred:
        :param y_true:
        :return:
        """
        n = y_pred.shape[0]
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return -1 * (-0.5 * n * self.const + 0.5 * n * (torch.digamma(ahat) - torch.log(bhat)) \
                     - 0.5 * (ahat/bhat) * ((y_pred - y_true) ** 2).sum())

    def kl(self):
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return (ahat - self.a0) * torch.digamma(ahat) - torch.lgamma(ahat) + gammaln(self.a0) + \
            self.a0 * (torch.log(bhat) - np.log(self.b0)) + ahat * (self.b0 - bhat) / bhat

    def get_noise_var(self):
        ahat = transform(self.ahat)
        bhat = transform(self.bhat)
        return (bhat / ahat).data.numpy()[0]


class GaussianNoiseFixedPrecision(torch.nn.Module, AbstractNoiseModel):
    """
        N(y_true | f(x, w), sigma_y**2); known sigma_y
    """

    def __init__(self, std_y=1., cuda=False):
        super(GaussianNoiseFixedPrecision, self).__init__()
        self.cuda = cuda
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))
        self.sigma_y = std_y

    def loss(self, y_pred=None, y_true=None):
        """
        computes -1 *  ln N (y_pred | y_true, sigma_y**2)
        :param y_pred:
        :param y_true:
        :return:
        """
        ll = -0.5 * self.const - np.log(self.sigma_y) - 0.5 * (1. / self.sigma_y ** 2) * ((y_pred - y_true) ** 2)
        return -ll.sum(dim=0)

    def get_noise_var(self):
        return self.sigma_y ** 2