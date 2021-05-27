import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch as torch


def make_data_gap(seed, data_count=100):
    import GPy
    npr.seed(0)
    x = np.hstack([np.linspace(-5, -2, int(data_count/2)), np.linspace(2, 5, int(data_count/2))])
    x = x[:, np.newaxis]
    k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    K = k.K(x)
    L = np.linalg.cholesky(K + 1e-5 * np.eye(data_count))

    # draw a noise free random function from a GP
    eps = np.random.randn(data_count)
    f = L @ eps


    # use a homoskedastic Gaussian noise model N(f(x)_i, \sigma^2). \sigma^2 = 0.1
    eps_noise =  np.sqrt(0.1) * np.random.randn(data_count)
    y = f + eps_noise
    y = y[:, np.newaxis]

    plt.plot(x, f, 'ko', ms=2)
    plt.plot(x, y, 'ro')
    plt.title("GP generated Data")
    plt.pause(1)
    return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(x), torch.FloatTensor(y)


def make_data_sine(seed, data_count=450):
    # fix the random seed
    np.random.seed(seed)
    noise_var = 0.1

    X = np.linspace(-4, 4, data_count)
    y = 1*np.sin(X) + np.sqrt(noise_var)*npr.randn(data_count)

    train_count = int (0.2 * data_count)
    idx = npr.permutation(range(data_count))
    X_train = X[idx[:train_count], np.newaxis ]
    X_test = X[ idx[train_count:], np.newaxis ]
    y_train = y[ idx[:train_count] ]
    y_test = y[ idx[train_count:] ]

    mu = np.mean(X_train, 0)
    std = np.std(X_train, 0)
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std
    mu = np.mean(y_train, 0)
    std = np.std(y_train, 0)
    # mu = 0
    # std = 1
    y_train = (y_train - mu) / std
    y_test = (y_test -mu) / std
    train_stats = dict()
    train_stats['mu'] = torch.FloatTensor([mu])
    train_stats['sigma'] = torch.FloatTensor([std])
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test),\
           train_stats