import os
import unittest
from unittest import TestCase

import pandas as pd
from tests.utils import create_train_test_prod_split, split
from uq360.utils.utils import UseTransformer

from uq360.algorithms.blackbox_metamodel.short_text_classification import ShortTextClassificationWrapper
import numpy as np
import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)


@unittest.skip("too long")
class TestShortTextClassification(TestCase):
    def _generate_mock_data(self, n_samples, n_classes, n_features):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42, class_sep=10)

    # Note, this test is expected to take some time (about 2 mins)
    def test_short_text_predictor(self):
        x, y = self.get_text_data()

        x_train, y_train, x_test, y_test, x_prod, y_prod = create_train_test_prod_split(x, y)

        obj = UseTransformer()
        x_train_encoded = obj.transform(X=x_train)
        x_test_encoded = obj.transform(X=x_test)
        x_prod_encoded = obj.transform(X=x_prod)
        x = np.concatenate((x_train_encoded, x_test_encoded, x_prod_encoded), axis=0)
        y = np.concatenate((y_train, y_test, y_prod), axis=0)

        # use the base model and grab the top confidence for every data point that we have
        model = self.train_model(x_train_encoded, y_train)

        x_proba = model.predict_proba(x)
        confs_sorted = np.sort(x_proba)
        top_confs = confs_sorted[:, -1]

        # find the median
        median = np.median(top_confs)
        # create two buckets
        less_than_median = np.where(top_confs < median)
        greater_than_median = np.where(top_confs >= median)

        x_train_new, y_train_new, x_test_new, y_test_new, prod_test_data, prod_test_label = split(x, y,
                                                                                                  less_than_median,
                                                                                                  greater_than_median,
                                                                                                  0.3)

        # train a new model using the training data created in the previous step
        model_trained_on_conf_based_split = self.train_model(x_train_new, y_train_new)
        # acc on test data
        acc = model_trained_on_conf_based_split.score(x_test_new, y_test_new)

        # acc on prod data
        acc = model_trained_on_conf_based_split.score(prod_test_data, prod_test_label)
        print("acc on prod", acc)

        p1 = ShortTextClassificationWrapper(base_model=model_trained_on_conf_based_split)

        p1.fit(x_train_new, y_train_new, x_test_new, y_test_new)

        y_mean, y_pred, y_score = p1.predict(prod_test_data)

        delta = abs(y_mean - acc * 100)
        self.assertTrue(delta <= 5)

    def train_model(self, x, y):
        """
        returns model object
        """
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        model.fit(x, y)

        return model

    def get_text_data(self):
        li_data = []
        li_labels = []
        li_len = []

        local_file = os.path.abspath(
            os.path.join(os.getcwd(), "..", "data", "text", "atis", "atis.train.w-intent.iob.csv"))

        df = pd.read_csv(local_file, index_col=None, header=0)
        li_data.append(df['example'])
        li_labels.append(df['intent'])

        frame = pd.concat(li_data, axis=0, ignore_index=True)
        npdata = frame.to_numpy()

        frame_labels = pd.concat(li_labels, axis=0, ignore_index=True)
        npdata_labels = frame_labels.to_numpy()

        return npdata, npdata_labels


if __name__ == '__main__':
    unittest.main()