import os
import unittest
from unittest import TestCase

from tests.test_utils import create_train_test_prod_split, split
from uq360.algorithms.classification_calibration import ClassificationCalibration
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.metrics import brier_score_loss
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
tf.get_logger().setLevel(logging.ERROR)


class TestCalibratedClassifier(TestCase):

    def setUp(self):
        self.num_classes = 2
        X, y = self._generate_mock_data(n_samples  = 10000,
             n_classes= self.num_classes, n_features = 8) 
    
        x_train, y_train, x_test, y_test, x_prod, y_prod = create_train_test_prod_split(X, y)

        self.x_test = x_test
        self.y_test = y_test
        self.x_prod = x_prod
        self.y_prod = y_prod
        self.model = self._train_model(x_train, y_train)



    def test_calibrated_classifier(self):    
        # Predictions for uncalibrated classifier
        y_pred = self.model.predict(self.x_prod)
        y_proba = self.model.predict_proba(self.x_prod)
        for method in ["isotonic", "sigmoid"]:
            # Estimator runs for fit_mode = 'probs'
            calib_model_probs = ClassificationCalibration(self.num_classes, 
                    fit_mode = 'probs', method = method)
            calib_model_probs.fit(self.model.predict_proba(self.x_test), self.y_test)
            res_probs = calib_model_probs.predict(self.model.predict_proba(self.x_prod))
            # Compare results with uncalibrated classifier
            print("Unaclibrate ",y_proba.shape)
            print("calibrated ",res_probs.y_prob.shape)

            assert_array_equal(y_pred, res_probs.y_pred)
            assert brier_score_loss(self.y_prod, y_proba[:,0]) >= brier_score_loss(
                self.y_prod, res_probs.y_prob[:,0])

            # Estimator runs for fit_mode = 'features'
            calib_model_feat = ClassificationCalibration(self.num_classes, fit_mode = 'features',
                    base_model_prediction_func = self.model.predict_proba,
                    method = method)

       

            calib_model_feat.fit(self.x_test, self.y_test)
            res_feat = calib_model_feat.predict(self.x_prod)


            assert_array_equal(y_pred, res_feat.y_pred)
            assert brier_score_loss(self.y_prod, y_proba[:,0]) >= brier_score_loss(
                self.y_prod, res_feat.y_prob[:,0])



    def _generate_mock_data(self, n_samples, n_classes, n_features):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42, class_sep=10)


    def _train_model(self, x, y):
        """
        returns pre-trained model object
        """
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        model.fit(x, y)
        return model


if __name__ == '__main__':
    unittest.main()