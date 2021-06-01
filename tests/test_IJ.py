import unittest

import autograd
import autograd.numpy as np
import numpy.random as npr
import scipy.optimize
from autograd import grad

from uq360.algorithms.infinitesimal_jackknife.infinitesimal_jackknife import InfinitesimalJackknife

npr.seed(42)

sigmoid = lambda x : 0.5 * (np.tanh(x / 2.) + 1)
get_num_train = lambda inputs : inputs.shape[0]
logistic_predictions = lambda params, inputs : sigmoid(np.dot(inputs, params))  # Outputs probability of a label being true according to logistic model.


class LogisticRegression:
    def __init__(self):
        self.params = None

    def set_parameters(self, params):
        self.params = params

    def predict(self, X):
        if self.params is not None:
            # Outputs probability of a label being true according to logistic model
            return np.atleast_2d(sigmoid(np.dot(X, self.params))).T
        else:
            raise RuntimeError("Params need to be fit before predictions can be made.")

    def compute_hessian(self, params_one, weights_one, inputs, targets):
        return autograd.hessian(self.loss, argnum=0)(params_one, weights_one, inputs, targets)

    def compute_jacobian(self, params_one, weights_one, inputs, targets):
        return autograd.jacobian(autograd.jacobian(self.loss, argnum=0), argnum=1)\
                                (params_one, weights_one, inputs, targets).squeeze()

    def loss(self, params, weights, inputs, targets):
        # Training loss is the negative log-likelihood of the training labels.
        preds = logistic_predictions(params, inputs)
        label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        return -np.sum(weights * np.log(label_probabilities + 1e-16))

    def fit(self, weights, init_params, inputs, targets, verbose=True):
        training_loss_fun = lambda params: self.loss(params, weights, inputs, targets)
        # Define a function that returns gradients of training loss using Autograd.
        training_gradient_fun = grad(training_loss_fun, 0)
        # optimize params
        print("Initial loss:", self.loss(init_params, weights, inputs, targets))
        # opt_params = sgd(training_gradient_fun, params, hyper=1, num_iters=5000, step_size=0.1)
        res = scipy.optimize.minimize(fun=training_loss_fun,
                                      jac=training_gradient_fun,
                                      x0=init_params,
                                      tol=1e-10,
                                      options={'disp': verbose})
        opt_params = res.x
        print("Trained loss:", self.loss(opt_params, weights, inputs, targets))
        self.params = opt_params
        return opt_params

    def get_test_acc(self, params, test_targets, test_inputs):
        preds = np.round(self.predict(test_inputs).T).astype(int)
        err = np.abs(test_targets - preds).sum()
        return 1 - err/ test_targets.shape[1]

    @staticmethod
    def synthetic_lr_data(N=10000, D=50):
        x = .5 * npr.randn(N, D)
        x_test = .5 * npr.randn(int(0.3 * N), D)
        w = npr.randn(D, 1)
        y = sigmoid(x @ w).ravel()
        y = npr.binomial(n=1, p=y)
        y_test = sigmoid(x_test @ w).ravel()
        # y_test = np.round(y_test)
        y_test = npr.binomial(n=1, p=y_test)
        return x, np.atleast_2d(y), x_test, np.atleast_2d(y_test)


class TestIJ(unittest.TestCase):

    def test_ij_exactj(self):
        # Create Model
        lr = LogisticRegression()
        inputs, targets, test_inputs, test_targets = lr.synthetic_lr_data()
        init_params = 1e-1 * np.random.randn(inputs.shape[1])

        # Standard MLE model fit
        weights_one = np.ones([1, get_num_train(inputs)])
        params_one = lr.fit(weights_one, init_params, inputs, targets)
        print("Maximum Likelihood Solution's Test Accuracy {0}".format(
            lr.get_test_acc(params_one, test_targets, test_inputs)))
        H = lr.compute_hessian(params_one, weights_one, inputs, targets)
        J = lr.compute_jacobian(params_one, weights_one, inputs, targets)

        # Approximate Jackknife
        config = {"alpha": 0.05, "resampling_strategy": 'jackknife'}
        ij = InfinitesimalJackknife(params_one, J, H, config)
        y_pred, y_lb, y_ub = ij.predict(test_inputs, lr)

        # compare approximate with exact jackknife for a few points
        weights = np.ones_like(weights_one)
        y_pred_exact = np.zeros([20, 1])
        y_pred_approx = np.zeros([20, 1])
        y_pred_mle = np.zeros([20, 1])
        for i in np.arange(20):
            weights[0, i] = 0
            params = lr.fit(weights, params_one, inputs, targets)
            lr.set_parameters(params)
            y_pred_exact[i] = lr.predict(inputs[i])
            ij_params = ij.ij(weights.squeeze())
            lr.set_parameters(ij_params)
            y_pred_approx[i] = lr.predict(inputs[i])
            lr.set_parameters(params_one)
            y_pred_mle[i] = lr.predict(inputs[i])
            weights[0, i] = 1

        print("Mean Absolute Deviation between approximate and exact jackknife: ",
              np.mean(np.abs(y_pred_approx - y_pred_exact)))
        assert np.allclose(y_pred_exact, y_pred_approx, atol=1e-6, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()






