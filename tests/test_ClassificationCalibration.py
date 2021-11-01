import os
import unittest
from unittest import TestCase

from tests.test_utils import create_train_test_prod_split, split
from uq360.algorithms.classification_calibration import ClassificationCalibration
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)


class TestCalibratedClassifier(TestCase):

    def setUp(self):
        self.num_classes = 4
        X, y = self._generate_mock_data(n_samples  = 10000,
             n_classes= self.num_classes, n_features = 8) 
    
        x_train, y_train, x_test, y_test, x_prod, y_prod = create_train_test_prod_split(X, y)

        self.x_test = x_test
        self.y_test = y_test
        self.x_prod = x_prod
        self.y_prod = y_prod
        self.model = self._train_model(x_train, y_train)

    def test_calibrated_classifier(self):
        
        # Check if calibrated classifier runs with probs method
        calib_model_probs = ClassificationCalibration(self.num_classes, 
                fit_mode = 'probs')

        calib_model_probs.fit(self.x_test, self.y_test)
        res_probs = calib_model_probs.predict(self.x_prod)

        # check if features fit_mode runs model run

        ## Gives an error right now
        calib_model_feat = ClassificationCalibration(self.num_classes, fit_mode = 'features',
                base_model_prediction_func = self._wrapper_for_fit)

        calib_model_feat.fit(self.x_test, self.y_test)
        res_feat = calib_model_feat.predict(self.x_prod)





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

    def _wrapper_for_fit(self):
        return self.model.predict_proba



if __name__ == '__main__':
    unittest.main()