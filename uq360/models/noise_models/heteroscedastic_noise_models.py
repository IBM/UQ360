import math

import numpy as np
import torch
from scipy.special import gammaln
from uq360.models.noise_models.noisemodel import AbstractNoiseModel
from torch.nn import Parameter

td = torch.distributions


def transform(a):
    return torch.log(1 + torch.exp(a))


class GaussianNoise(torch.nn.Module, AbstractNoiseModel):
    """
        N(y_true | f_\mu(x, w), f_\sigma^2(x, w))
    """

    def __init__(self, cuda=False):
        super(GaussianNoise, self).__init__()
        self.cuda = cuda
        self.const = torch.log(torch.FloatTensor([2 * math.pi]))

    def loss(self, y_true=None, mu_pred=None, log_var_pred=None, reduce_mean=True):
        """
        computes -1 *  ln N (y_true | mu_pred, softplus(log_var_pred))
        :param y_true:
        :param mu_pred:
        :param log_var_pred:

        :return:
        """
        var_pred = transform(log_var_pred)
        ll = -0.5 * self.const - 0.5 * torch.log(var_pred) - 0.5 * (1. / var_pred) * ((mu_pred - y_true) ** 2)
        if reduce_mean:
            return -ll.mean(dim=0)
        else:
            return -ll.sum(dim=0)

    def get_noise_var(self, log_var_pred):
        return transform(log_var_pred)


