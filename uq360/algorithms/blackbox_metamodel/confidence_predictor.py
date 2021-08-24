import inspect
from collections import namedtuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from uq360.algorithms.blackbox_metamodel.predictors.predictor_driver import PredictorDriver
from uq360.algorithms.posthocuq import PostHocUQ


class ConfidencePredictor(PostHocUQ):

    def __init__(self, base_model=None):
        super(ConfidencePredictor).__init__(base_model)
        self.client_model = base_model
        self.performance_predictor = "confidence"
        self.calib = 'isotonic_regression'
        self.fitted = False
        self.driver = PredictorDriver(self.performance_predictor,
                                      base_model=self.client_model,
                                      pointwise_features=None,
                                      batch_features=None,
                                      blackbox_features=None,
                                      use_whitebox=True,
                                      use_drift_classifier=True,
                                      calibrator=self.calib)

    def fit(self, x_train, y_train, x_test, y_test, test_predicted_probabilities=None):
        self.driver.fit(x_train, y_train, x_test, y_test, test_predicted_probabilities=test_predicted_probabilities)
        self.fitted = True

    def _process_pretrained_model(self, x, y_hat):
        raise NotImplementedError

    def predict(self, x, return_predictions=False, predicted_probabilities=None):
        if not self.fitted:
            raise Exception("Untrained Predictor: fit() method needs to be called before predicting.")

        predictions = self.driver.predict(x, predicted_probabilities=predicted_probabilities)

        output = {'predicted_accuracy': predictions['accuracy'], 'uncertainty': predictions['uncertainty']}
        if 'error' in predictions:
            output['error'] = predictions['error']

        if return_predictions:
            output['predictions_per_datapoint'] = predictions['pointwise_confidences']

        return output
