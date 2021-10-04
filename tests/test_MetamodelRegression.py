import unittest
from unittest import TestCase

import numpy as np


class TestMetamodelRegression(TestCase):
    def _generate_mock_data(self, n_samples, n_features):
        from sklearn.datasets import make_regression
        return make_regression(n_samples, n_features, random_state=42)

    def test_create_gbr_model_by_name(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_regression as bb
        X, y = self._generate_mock_data(200, 3)
        m = bb.MetamodelRegression(base_model='gbr', meta_model='gbr')
        m.fit(X, y)
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        print(yhat, yhat_lb, yhat_ub)
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -148.85602688) and np.isclose(yhat_ub, 138.09324391))


    def test_create_gbr_model_from_class(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_regression as bb
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = self._generate_mock_data(200, 3)
        m = bb.MetamodelRegression(base_model=GradientBoostingRegressor, meta_model=GradientBoostingRegressor)
        m.fit(X, y)
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -148.85602688) and np.isclose(yhat_ub, 138.09324391))


    def test_create_gbr_model_from_instance(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_regression as bb
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = self._generate_mock_data(200, 3)
        meta_config = {'loss': 'quantile', 'alpha': 0.95, 'n_estimators': 300, 'max_depth': 10,
                                    'learning_rate': 0.001, 'min_samples_leaf': 10, 'min_samples_split': 10,
                                    'random_state': 42}
        gbr_meta = GradientBoostingRegressor(**meta_config)
        m = bb.MetamodelRegression(base_model='gbr', meta_model=gbr_meta)
        m.fit(X, y)
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -148.85602688) and np.isclose(yhat_ub, 138.09324391))
        base_config = {'loss': 'ls', 'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.001,
         'min_samples_leaf': 10, 'min_samples_split': 10, 'random_state': 42}
        gbr_base = GradientBoostingRegressor(**base_config)
        m = bb.MetamodelRegression(base_model=gbr_base, meta_model=gbr_meta)
        m.fit(X, y)
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -148.85602688) and np.isclose(yhat_ub, 138.09324391))

    def test_create_lr_model_from_instance(self):
        from uq360.algorithms.blackbox_metamodel import metamodel_regression as bb
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        X, y = self._generate_mock_data(200, 3)
        base_config = {'loss': 'ls', 'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.001,
                       'min_samples_leaf': 10, 'min_samples_split': 10, 'random_state': 42}
        meta_config = {'fit_intercept': True, 'copy_X': True}
        lr_meta = LinearRegression(**meta_config)
        m = bb.MetamodelRegression(base_model=GradientBoostingRegressor, meta_model=lr_meta,
                                   base_config=base_config, meta_config=meta_config)
        m.fit(X, y)
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -66.49968693) and np.isclose(yhat_ub, 55.73690395))
        # now use the created base as 'external'
        mdl = m.base_model
        del m
        m = bb.MetamodelRegression(base_model=mdl, meta_model=lr_meta,
                                   base_config=base_config, meta_config=meta_config)
        m.fit(X, y, base_is_prefitted=True)
        # result must be identical with previous
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -66.49968693) and np.isclose(yhat_ub, 55.73690395))
        # re-fit with user-supplied meta data
        m.fit(None, None, base_is_prefitted=True, meta_train_data=(X, y))
        yhat, yhat_lb, yhat_ub = m.predict(X[0:1, :])
        assert(np.isclose(yhat, -5.38139149) and np.isclose(yhat_lb, -66.49362395) and np.isclose(yhat_ub, 55.73084097))


if __name__ == '__main__':
    unittest.main()