from collections import namedtuple

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

from uq360.utils.misc import DummySklearnEstimator
from uq360.algorithms.posthocuq import PostHocUQ


class ClassificationCalibration(PostHocUQ):
    """Post hoc calibration of classification models. Currently wraps `CalibratedClassifierCV` from sklearn and allows
    non-sklearn models to be calibrated.

    """
    def __init__(self, num_classes, fit_mode="features", method='isotonic', base_model_prediction_func=None):
        """

        Args:
            num_classes: number of classes.
            fit_mode: features or probs. If probs the `fit` and `predict` operate on the base models probability scores,
                useful when these are precomputed.
            method: isotonic or sigmoid.
            base_model_prediction_func: the function that takes in the input features and produces base model's
                probability scores. This is ignored when operating in `probs` mode.
        """
        super(ClassificationCalibration).__init__()
        if fit_mode == "probs":
            # In this case, the fit assumes that it receives the probability scores of the base model.
            # create a dummy estimator
            self.base_model = DummySklearnEstimator(num_classes, lambda x: x)
        else:
            self.base_model = DummySklearnEstimator(num_classes, base_model_prediction_func)
        self.method = method

    def get_params(self, deep=True):
        return {"num_classes": self.num_classes, "fit_mode": self.fit_mode, "method": self.method,
                "base_model_prediction_func": self.base_model_prediction_func}

    def _process_pretrained_model(self, base_model):
        return base_model

    def fit(self, X, y):
        """ Fits calibration model using the provided calibration set.

        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples, n_classes).
                Features vectors of the training data or the probability scores from the base model.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """

        self.base_model.label_encoder_ = LabelEncoder().fit(y)
        self.calib_model = CalibratedClassifierCV(base_estimator=self.base_model,
                                                  cv="prefit",
                                                  method=self.method)
        self.calib_model.fit(X, y)

        return self

    def predict(self, X):
        """
        Obtain calibrated predictions for the test points.

        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples, n_classes).
                Features vectors of the training data or the probability scores from the base model.

        Returns:
            namedtuple: A namedtupe that holds

            y_pred: ndarray of shape (n_samples,)
                Predicted labels of the test points.
            y_prob: ndarray of shape (n_samples, n_classes)
                Predicted probability scores of the classes.

        """
        y_prob = self.calib_model.predict_proba(X)
        if len(np.shape(y_prob)) == 1:
            y_pred_labels = y_prob > 0.5

        else:
            y_pred_labels = np.argmax(y_prob, axis=1)

        Result = namedtuple('res', ['y_pred', 'y_prob'])
        res = Result(y_pred_labels, y_prob)

        return res
