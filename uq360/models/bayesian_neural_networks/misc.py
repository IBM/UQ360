import numpy as np
import torch
from uq360.models.noise_models.homoscedastic_noise_models import GaussianNoiseFixedPrecision

def compute_test_ll(y_test, y_pred_samples, std_y=1.):
    """
    Computes test log likelihoods = (1 / Ntest) * \sum_n p(y_n | x_n, D_train)
    :param y_test: True y
    :param y_pred_samples: y^s = f(x_test, w^s); w^s ~ q(w). S x Ntest, where S is the number of samples
    q(w) is either a trained variational posterior or an MCMC approximation to p(w | D_train)
    :param std_y: True std of y (assumed known)
    """
    S, _ = y_pred_samples.shape
    noise = GaussianNoiseFixedPrecision(std_y=std_y)
    ll = noise.loss(y_pred=y_pred_samples, y_true=y_test.unsqueeze(dim=0), reduce_sum=False)
    ll = torch.logsumexp(ll, dim=0) - np.log(S)  # mean over num samples
    return torch.mean(ll)  # mean over test points


