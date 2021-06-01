import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})

from copy import deepcopy

import numpy as np
import numpy.random as npr


def make_batches(n_data, batch_size):
    return [slice(i, min(i+batch_size, n_data)) for i in range(0, n_data, batch_size)]


def generate_regression_data(seed, data_count=500):
    """
    Generate data from a noisy sine wave.
    :param seed: random number seed
    :param data_count: number of data points.
    :return:
    """
    np.random.seed(seed)
    noise_var = 0.1

    x = np.linspace(-4, 4, data_count)
    y = 1*np.sin(x) + np.sqrt(noise_var)*npr.randn(data_count)

    train_count = int (0.2 * data_count)
    idx = npr.permutation(range(data_count))
    x_train = x[idx[:train_count], np.newaxis ]
    x_test = x[ idx[train_count:], np.newaxis ]
    y_train = y[ idx[:train_count] ]
    y_test = y[ idx[train_count:] ]

    mu = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
    mu = np.mean(y_train, 0)
    std = np.std(y_train, 0)
    y_train = (y_train - mu) / std
    train_stats = dict()
    train_stats['mu'] = mu
    train_stats['sigma'] = std

    return x_train, y_train, x_test, y_test, train_stats


def form_D_for_auucc(yhat, zhatl, zhatu):
    # a handy routine to format data as needed by the UCC fit() method
    D = np.zeros([yhat.shape[0], 3])
    D[:, 0] = yhat.squeeze()
    D[:, 1] = zhatl.squeeze()
    D[:, 2] = zhatu.squeeze()
    return D


def fitted_ucc_w_nullref(y_true, y_pred_mean, y_pred_lower, y_pred_upper):
    """
    Instantiates an UCC object for the target predictor plus a 'null' (constant band) reference
    :param y_pred_lower:
    :param y_pred_mean:
    :param y_pred_upper:
    :param y_true:
    :return: ucc object fitted for two systems: target  + null reference
    """
    # form matrix for ucc:
    X_for_ucc = form_D_for_auucc(y_pred_mean.squeeze(),
                                 y_pred_mean.squeeze() - y_pred_lower.squeeze(),
                                 y_pred_upper.squeeze() - y_pred_mean.squeeze())
    # form matrix for a 'null' system (constant band)
    X_null = deepcopy(X_for_ucc)
    X_null[:,1:] = np.std(y_pred_mean)  # can be set to any other constant (no effect on AUUCC)
    # create an instance of ucc and fit data
    from uq360.metrics.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
    u = ucc()
    u.fit([X_for_ucc, X_null], y_true.squeeze())
    return u


def make_sklearn_compatible_scorer(task_type, metric, greater_is_better=True, **kwargs):
    """

    Args:
        task_type: (str) regression or classification.
        metric: (str): choice of metric can be one of these - [aurrrc, ece, auroc, nll, brier, accuracy] for
            classification and ["rmse", "nll", "auucc_gain", "picp", "mpiw", "r2"] for regression.
        greater_is_better: is False the scores are negated before returning.
        **kwargs: additional arguments specific to some metrics.

    Returns:
        sklearn compatible scorer function.

    """

    from uq360.metrics.classification_metrics import compute_classification_metrics
    from uq360.metrics.regression_metrics import compute_regression_metrics

    def sklearn_compatible_score(model, X, y_true):
        """

        Args:
            model: The model being scored. Currently uq360 and sklearn models are supported.
            X: Input features.
            y_true: ground truth values for the target.

        Returns:
            Computed score of the model.

        """

        from uq360.algorithms.builtinuq import BuiltinUQ
        from uq360.algorithms.posthocuq import PostHocUQ
        if isinstance(model, BuiltinUQ) or isinstance(model, PostHocUQ):
            # uq360 models
            if task_type == "classification":
                score = compute_classification_metrics(
                    y_true=y_true,
                    y_prob=model.predict(X).y_prob,
                    option=metric,
                    **kwargs
                )[metric]
            elif task_type == "regression":
                y_mean, y_lower, y_upper = model.predict(X)
                score = compute_regression_metrics(
                    y_true=y_true,
                    y_mean=y_mean,
                    y_lower=y_lower,
                    y_upper=y_upper,
                    option=metric,
                    **kwargs
                )[metric]
            else:
                raise NotImplementedError

        else:
            # sklearn models
            if task_type == "classification":
                score = compute_classification_metrics(
                    y_true=y_true,
                    y_prob=model.predict_proba(X),
                    option=metric,
                    **kwargs
                )[metric]
            else:
                if metric in ["rmse", "r2"]:
                    score = compute_regression_metrics(
                        y_true=y_true,
                        y_mean=model.predict(X),
                        y_lower=None,
                        y_upper=None,
                        option=metric,
                        **kwargs
                    )[metric]
                else:
                    raise NotImplementedError("{} is not supported for sklearn regression models".format(metric))

        if not greater_is_better:
            score = -score
        return score
    return sklearn_compatible_score


class DummySklearnEstimator(ABC):
    def __init__(self, num_classes, base_model_prediction_fn):
        self.base_model_prediction_fn = base_model_prediction_fn
        self.classes_ = [i for i in range(num_classes)]

    def fit(self):
        pass

    def predict_proba(self, X):
        return self.base_model_prediction_fn(X)
