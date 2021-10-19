from builtins import range
import autograd.numpy as np


def adam(grad, x, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, polyak=False):
    """Adapted from autograd.misc.optimizers"""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g, polyak)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x