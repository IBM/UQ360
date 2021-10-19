import unittest
from unittest import TestCase

import numpy as np


class TestMetamodelClassification(TestCase):
    def _generate_mock_data(self, n_samples, n_classes, n_features):
        from sklearn.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_features, n_redundant=0, random_state=42)

    def test_create_gbm_model_by_name(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_classification as bb
        X, y = self._generate_mock_data(200, 4, 3)
        m = bb.MetamodelClassification(base_model='gbm', meta_model='lr')
        m.fit(X, y)
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.52529918, 0.23764194, 0.93118627, 0.80335416, 0.95338512], rtol=1.e-03)))


    def test_create_gbm_model_from_class(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_classification as bb
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = self._generate_mock_data(200, n_classes=4, n_features=3)
        m = bb.MetamodelClassification(base_model=GradientBoostingClassifier, meta_model=LogisticRegression)
        m.fit(X, y)
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.52529918, 0.23764194, 0.93118627, 0.80335416, 0.95338512], rtol=1.e-03)))


    def test_create_gbm_model_from_instance(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_classification as bb
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = self._generate_mock_data(200, n_classes=4, n_features=3)
        base_config = {'n_estimators': 300, 'max_depth': 10,
                                    'learning_rate': 0.001, 'min_samples_leaf': 10, 'min_samples_split': 10,
                                    'random_state': 42}
        meta_config = {'penalty': 'l1', 'C': 1, 'solver': 'liblinear', 'random_state': 42}
        gbm_base = GradientBoostingClassifier(**base_config)
        # just base is an instance
        m = bb.MetamodelClassification(base_model=gbm_base, meta_model='lr')
        m.fit(X, y)
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.52529918, 0.23764194, 0.93118627, 0.80335416, 0.95338512], rtol=1.e-03)))
        # both base and meta are instances
        lr_meta = LogisticRegression(**meta_config)
        m = bb.MetamodelClassification(base_model=gbm_base, meta_model=lr_meta)
        m.fit(X, y)
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.52529918, 0.23764194, 0.93118627, 0.80335416, 0.95338512], rtol=1.e-03)))

    def test_use_prefitted_base_model(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_classification as bb
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = self._generate_mock_data(200, n_classes=4, n_features=3)
        base_config = {'n_estimators': 300, 'max_depth': 10,
                                    'learning_rate': 0.001, 'min_samples_leaf': 10, 'min_samples_split': 10,
                                    'random_state': 42}
        meta_config = {'penalty': 'l1', 'C': 1, 'solver': 'liblinear', 'random_state': 42}
        gbm_base = GradientBoostingClassifier(**base_config)
        lr_meta = LogisticRegression(**meta_config)
        gbm_base.fit(X, y)
        m = bb.MetamodelClassification(base_model=gbm_base, meta_model=lr_meta)
        m.fit(X, y, base_is_prefitted=True, meta_fraction=.99)
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.70175115, 0.93548402, 0.95223033, 0.90373872, 0.87709154], rtol=1.e-03)))
        m.fit(None, None, base_is_prefitted=True, meta_train_data=(X, y))
        y_pred, y_prob = m.predict(X[0:5, :])
        assert (all(y_pred == [3, 3, 0, 1, 0]))
        assert (all(np.isclose(y_prob, [0.6894514 , 0.9355004 , 0.95257614, 0.89933285, 0.87657669], rtol=1.e-03)))


if __name__ == '__main__':
    unittest.main()