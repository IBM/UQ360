import autograd
import autograd.numpy as np
import numpy.random as npr
import scipy.optimize

sigmoid = lambda x: 0.5 * (np.tanh(x / 2.) + 1)
get_num_train = lambda inputs: inputs.shape[0]
logistic_predictions = lambda params, inputs: sigmoid(np.dot(inputs, params))


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

    def loss(self, params, weights, inputs, targets):
        # Training loss is the negative log-likelihood of the training labels.
        preds = logistic_predictions(params, inputs)
        label_probabilities = preds * targets + (1 - preds) * (1 - targets)
        return -np.sum(weights * np.log(label_probabilities + 1e-16))

    def fit(self, weights, init_params, inputs, targets, verbose=True):
        training_loss_fun = lambda params: self.loss(params, weights, inputs, targets)
        # Define a function that returns gradients of training loss using Autograd.
        training_gradient_fun = autograd.grad(training_loss_fun, 0)
        # optimize params
        if verbose:
            print("Initial loss:", self.loss(init_params, weights, inputs, targets))
        # opt_params = sgd(training_gradient_fun, params, hyper=1, num_iters=5000, step_size=0.1)
        res = scipy.optimize.minimize(fun=training_loss_fun,
                                      jac=training_gradient_fun,
                                      x0=init_params,
                                      tol=1e-6,
                                      options={'disp': verbose})
        opt_params = res.x
        if verbose:
            print("Trained loss:", self.loss(opt_params, weights, inputs, targets))
        self.params = opt_params
        return opt_params

    def get_test_acc(self, params, test_targets, test_inputs):
        preds = np.round(self.predict(test_inputs).T).astype(np.int)
        err = np.abs(test_targets - preds).sum()
        return 1 - err/ test_targets.shape[1]

    #### Required for IJ computation ###
    def compute_hessian(self, params_one, weights_one, inputs, targets):
        return autograd.hessian(self.loss, argnum=0)(params_one, weights_one, inputs, targets)

    def compute_jacobian(self, params_one, weights_one, inputs, targets):
        return autograd.jacobian(autograd.jacobian(self.loss, argnum=0), argnum=1)\
                                (params_one, weights_one, inputs, targets).squeeze()
    ###################################################

    @staticmethod
    def synthetic_lr_data(N=10000, D=10):
        x = 2. * npr.randn(N, D)
        x_test = 1. * npr.randn(int(0.3 * N), D)
        w = npr.randn(D, 1)
        y = sigmoid((x @ w)).ravel()
        y = npr.binomial(n=1, p=y) # corrupt labels
        y_test = sigmoid(x_test @ w).ravel()
        # y_test = np.round(y_test)
        y_test = npr.binomial(n=1, p=y_test)
        return x, np.atleast_2d(y), x_test, np.atleast_2d(y_test)
