import unittest
from unittest import TestCase

import numpy as np


class TestBlackBoxMetamodelClassification(TestCase):
    def _generate_mock_data(self, n_samples, n_classes, n_features):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42)

    def create_train_test_prod_split(self, x, y, test_size=0.25):
        """
        returns x_train, y_train, x_test, y_test, x_prod, y_prod
        """
        from sklearn.model_selection import StratifiedKFold, train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.25,
                                                            random_state=42)

        x_test, x_prod, y_test, y_prod = train_test_split(x_test, y_test,
                                                          test_size=0.25,
                                                          random_state=42)

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_prod.shape, y_prod.shape)

        print("Training data size:", x_train.shape)
        print("Test data size:", x_test.shape)
        print("Prod data size:", x_prod.shape)

        return x_train, y_train, x_test, y_test, x_prod, y_prod

    def test_structured_data_pred(self):
        X, y = self._generate_mock_data(10000, n_classes=2, n_features=10)
        x_train, y_train, x_test, y_test, x_prod, y_prod = self.create_train_test_prod_split(X, y)
        #
        from sklearn.ensemble import RandomForestClassifier
        from uq360.algorithms.blackbox_metamodel.structured_data_predictor import StructuredDataPredictorWrapper

        # rf = RandomForestClassifier()
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=8, max_features='auto', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=576,
                                     n_jobs=None, oob_score=False, random_state=42, verbose=0,
                                     warm_start=False)

        rf.fit(x_train, y_train)

        acc_on_test = rf.score(x_test, y_test)
        print('Accuracy on test: ', acc_on_test * 100)

        acc_on_prod = rf.score(x_prod, y_prod)
        print('Accuracy on prod: ', acc_on_prod * 100)



        p = StructuredDataPredictorWrapper(base_model=rf)
        p.fit(x_train, y_train, x_test, y_test)

        # predict the model's accuracy on production/unlabeled data
        prediction = p.predict(x_prod)

        print(prediction)

if __name__ == '__main__':
    unittest.main()