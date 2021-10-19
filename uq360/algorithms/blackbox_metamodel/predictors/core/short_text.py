# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#


from collections import Counter
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor
from uq360.utils.hpo_search import CustomRandomSearch
from uq360.utils.calibrators.calibrator import Calibrator
import logging
# import numpy as np
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from .performance_predictor import PerfPredictor
# from ..hpo_search import CustomRandomSearch
# from ..calibrators.calibrator import Calibrator
import logging

logger = logging.getLogger(__name__)


"""
Performance predictor for short text data. It is based on an ensemble of meta-models: 
one mlp metamodel, one GBM metamodel, and one SVM metamodel. This performance predictor does not have a method 
to quantify its own uncertainty, so the uncertainty values are zero.  
"""
class TextEnsemblePredictor(PerfPredictor):
    def __init__(self, calibrator="shift"):
        self.metamodels_considered = ["svm", "gbm", "mlp"]
        self.metamodels = {}
        self.metamodel_calibrators = {}

        self.return_all_true = False
        self.return_all_false = False
        self.x_test = None
        self.y_test = None
        self.random_state = 42
        self._object_registry = {}
        self.fit_status = False

        # A dictionary to stash any whitebox features the prediction has that can be used in the uncertainty model
        self.whitebox_features = {}

        logger.info("Calibrator: %s", calibrator)
        if calibrator is None:
            self.metamodel_calibrators = None
        else:
            for metamodel in self.metamodels_considered:
                self.metamodel_calibrators[metamodel] = Calibrator.instance(calibrator)


    @classmethod
    def name(cls):
        return ('text_ensemble')

    def fit(self, x_test_unprocessed, x_test, y_test):

        self.x_test = x_test
        self.y_test = y_test

        x_test = x_test.values

        # Don't split off test set for calibration if calibrator = None
        if self.metamodel_calibrators is not None:
            try:
                x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test,
                                                                test_size=0.25,
                                                                stratify=y_test,
                                                            random_state=self.random_state)

            except Exception as e:
                # sometimes it may not be possible to stratify - when all the predictors are correct or incorrect.
                # fall back to regular train test split and these conditions will be handled downstream
                x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.2,
                                                                random_state=self.random_state)

        else:
            x_dev = x_test
            y_dev = y_test

        if len(np.unique(y_dev)) == 1:

            if 1 in y_dev:
                self.return_all_true = True
                print(
                    'The base model has an accuracy of 100 percent on the test set. Return predictions of only 100 percent')
                self.fit_status = True
                return
            else:
                self.return_all_false = True
                print(
                    'The base model has an accuracy of 0 percent on the test set. Return predictions of only 0 percent')
                self.fit_status = True
                return

        # Balance datasets
        x_dev, y_dev = self._balance_data(x_dev, y_dev)

        mlp_parameters =  {
                "hidden_layer_sizes": [(100,),
                                       (100, 100, 100,),
                                       (300, 300,),
                                       (400, 300, 200, 100,)],
                "activation": ['logistic', 'relu'],
                "early_stopping": [True],
                "learning_rate": ['constant', 'adaptive'],
                "alpha": [0.00001, 0.0001, 0.001]
            }


        svm_parameters = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'linear', 'poly','sigmoid']
        }

        gbm_parameters = {
            "loss": ["deviance"],
            "learning_rate": [0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.005, 0.01, 5),
            "min_samples_leaf": np.linspace(0.0005, 0.001, 5),
            "max_leaf_nodes": list(range(3, 12, 2)),
            "max_features": ["log2", "sqrt"],
            "subsample": np.linspace(0.3, 0.9, 6),
            "n_estimators": range(100, 401, 50)
        }

        randomized_params = {
            "n_iter": 20,
            "scoring": "f1",
            "n_jobs": -1,
            "cv": StratifiedKFold(n_splits=3, shuffle=True),
            "verbose": 0,
            "return_train_score": True,
            "progress_bar": False,
            "random_state": self.random_state}
        classifier1 = GradientBoostingClassifier()
        gbm_classifier = CustomRandomSearch(classifier1, gbm_parameters, **randomized_params)


        gbm = None

        if 'gbm' in self.metamodels_considered:
            gbm_classifier.fit(x_dev, y_dev)
            gbm = gbm_classifier.best_estimator_
            self.metamodels["gbm"] = gbm
            logger.info("Building GBM Model is complete")

        classifier2 = MLPClassifier()
        mlp_classifier = CustomRandomSearch(classifier2, mlp_parameters, **randomized_params)

        mlp = None

        if 'mlp' in self.metamodels_considered:
            mlp_classifier.fit(x_dev, y_dev)
            mlp = mlp_classifier.best_estimator_

            self.metamodels["mlp"] = mlp
            logger.info("Building MLP Model is complete")


        classifier3 = SVC(probability=True,max_iter=10000)
        svm_classifier = CustomRandomSearch(classifier3, svm_parameters, **randomized_params)
        svm = None

        if 'svm' in self.metamodels_considered:
            svm_classifier.fit(x_dev, y_dev)
            svm = svm_classifier.best_estimator_

            logger.info("Building SVM Model is complete")
            self.metamodels["svm"] = svm



        # If calibrator is not None, fit
        if self.metamodel_calibrators is not None:
            if len(np.unique(y_test)) == 1:
                if 1 in y_test:
                    self.return_all_true = True
                    print(
                        'The base model has an accuracy of 100 percent on the test set. Return predictions of only 100 percent')
                    self.fit_status = True
                    return
                else:
                    self.return_all_false = True
                    print(
                        'The base model has an accuracy of 0 percent on the test set. Return predictions of only 0 percent')
                    self.fit_status = True
                    return

            logger.info("Metamodels considered %s", self.metamodels_considered)
            for mm in self.metamodels_considered:
                model = self.metamodels[mm]
                preds = model.predict_proba(x_test)
                preds = preds[:, 1]
                self.metamodel_calibrators[mm].fit(preds, y_test)

        self.fit_status = True

    def predict(self, X_unprocessed, X):

        X = X.values
        assert self.fit_status
        if self.return_all_true:
            preds = 0.99999999 * np.ones(X.shape[0])
            output = {'confidences': preds, 'uncertainties': np.zeros(preds.shape)}
            return output

        if self.return_all_false:
            preds = 0 * np.ones(X.shape[0])
            output = {'confidences': preds, 'uncertainties': np.zeros(preds.shape)}
            return output

        confidences = []
        for mm in self.metamodels_considered:
            logger.info("Predicting against metamodel %s", mm)
            model = self.metamodels[mm]
            preds = model.predict_proba(X)
            preds = preds[:, 1]

            if self.metamodel_calibrators:
                out = self.metamodel_calibrators[mm].predict(preds)
                out = list(map(lambda x: 0 if x < 0 else x, out))
                out = list(map(lambda x: 1 if x > 1 else x, out))
                confidences.append(out)
            else:
                confidences.append(preds)
        preds = confidences
        confidences = np.maximum(np.mean(preds, axis=0), 0.0)
        output = {'confidences': confidences, 'uncertainties': np.zeros(confidences.shape)}
        return output


    def save(self, output_location):
        pass

    def load(self, input_location):
        pass

#
#
# import numpy as np
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor
# from uq360.utils.hpo_search import CustomRandomSearch
# from uq360.utils.calibrators.calibrator import Calibrator
# import logging
#
# logger = logging.getLogger(__name__)
#
# """
# This version of the text ensemble predictor does an "inner" and "outer calibration".
# Note: For calibration, we just set aside one small subset of data and reuse it for inner and outer calibration
#
# Logic:
# fit()
# Take every meta model, create a calib object per meta model (N objects). predict against the meta model and fit() one calibrator per meta model. (inner calib)
# As you create one calib per metamodel, save the predictions and obtain the mean confidences. Use this to fit a "master calibrator" (outer calib)
#
# predict()
# obtain confidences against every single meta model. pass the confidences to the calibrator's predict().
# grab the mean of confidences that come from N calibrators (inner calib)
# pass the mean of confidences to the "master calibrator" and predict() again. (outer calib)
#
# """
#
#
# class TextEnsembleV2Predictor(PerfPredictor):
#
#     def __init__(self, calibrator="isotonic_regression"):
#         self.metamodels_considered = ["svm", "gbm", "mlp"]
#         self.metamodels = {}
#         self.metamodel_calibrators = {}
#
#         self.return_all_true = False
#         self.return_all_false = False
#         self.x_test = None
#         self.y_test = None
#         self.random_state = 42
#         self._object_registry = {}
#         self.fit_status = False
#
#         # A dictionary to stash any whitebox features the prediction has that can be used in the uncertainty model
#         self.whitebox_features = {}
#
#         logger.info("Calibrator: %s", calibrator)
#         if calibrator is None:
#             self.metamodel_calibrators = None
#         else:
#             for metamodel in self.metamodels_considered:
#                 self.metamodel_calibrators[metamodel] = Calibrator.instance(calibrator)
#
#         if calibrator is None:
#             self.calibrator = None
#         else:
#             self.calibrator = Calibrator.instance(calibrator)
#
#     @classmethod
#     def name(cls):
#         return ('text_ensemble')
#
#     def fit(self, x_test_unprocessed, x_test, y_test):
#         self.x_test = x_test
#         self.y_test = y_test
#
#         x_test = x_test.values
#
#         # Don't split off test set for calibration if calibrator = None
#         if self.metamodel_calibrators is not None:
#             try:
#                 x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test,
#                                                                 test_size=0.25,
#                                                                 stratify=y_test,
#                                                                 random_state=self.random_state)
#
#             except Exception as e:
#                 # sometimes it may not be possible to stratify - when all the predictors are correct or incorrect.
#                 # fall back to regular train test split and these conditions will be handled downstream
#                 x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.2,
#                                                                 random_state=self.random_state)
#         else:
#             x_dev = x_test
#             y_dev = y_test
#
#         if len(np.unique(y_dev)) == 1:
#             if 1 in y_dev:
#                 self.return_all_true = True
#                 print(
#                     'The base model has an accuracy of 100 percent on the test set. Return predictions of only 100 percent')
#                 self.fit_status = True
#                 return
#             else:
#                 self.return_all_false = True
#                 print(
#                     'The base model has an accuracy of 0 percent on the test set. Return predictions of only 0 percent')
#                 self.fit_status = True
#                 return
#         # Balance datasets
#         x_dev, y_dev = self._balance_data(x_dev, y_dev)
#         mlp_parameters = {
#             "hidden_layer_sizes": [(100,),
#                                    (100, 100, 100,),
#                                    (300, 300,),
#                                    (400, 300, 200, 100,)],
#             "activation": ['logistic', 'relu'],
#             "early_stopping": [True],
#             "learning_rate": ['constant', 'adaptive'],
#             "alpha": [0.00001, 0.0001, 0.001]
#         }
#
#         svm_parameters = {
#             'C': [0.1, 1, 10, 100, 1000],
#             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#             'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
#         }
#
#         gbm_parameters = {
#             "loss": ["deviance"],
#             "learning_rate": [0.1, 0.15, 0.2],
#             "min_samples_split": np.linspace(0.005, 0.01, 5),
#             "min_samples_leaf": np.linspace(0.0005, 0.001, 5),
#             "max_leaf_nodes": list(range(3, 12, 2)),
#             "max_features": ["log2", "sqrt"],
#             "subsample": np.linspace(0.3, 0.9, 6),
#             "n_estimators": range(100, 401, 50)
#         }
#
#         randomized_params = {
#             "n_iter": 20,
#             "scoring": "f1",
#             "n_jobs": -1,
#             "cv": StratifiedKFold(n_splits=3, shuffle=True),
#             "verbose": 0,
#             "return_train_score": True,
#             "progress_bar": False,
#             "random_state": self.random_state}
#         classifier1 = GradientBoostingClassifier()
#         gbm_classifier = CustomRandomSearch(classifier1, gbm_parameters, **randomized_params)
#
#         gbm = None
#
#         if 'gbm' in self.metamodels_considered:
#             gbm_classifier.fit(x_dev, y_dev)
#             gbm = gbm_classifier.best_estimator_
#             self.metamodels["gbm"] = gbm
#             logger.info("Building GBM Model is complete")
#
#         classifier2 = MLPClassifier()
#         mlp_classifier = CustomRandomSearch(classifier2, mlp_parameters, **randomized_params)
#
#         mlp = None
#
#         if 'mlp' in self.metamodels_considered:
#             mlp_classifier.fit(x_dev, y_dev)
#             mlp = mlp_classifier.best_estimator_
#
#             self.metamodels["mlp"] = mlp
#             logger.info("Building MLP Model is complete")
#
#         classifier3 = SVC(probability=True, max_iter=10000)
#         svm_classifier = CustomRandomSearch(classifier3, svm_parameters, **randomized_params)
#         svm = None
#
#         if 'svm' in self.metamodels_considered:
#             svm_classifier.fit(x_dev, y_dev)
#             svm = svm_classifier.best_estimator_
#
#             logger.info("Building SVM Model is complete")
#             self.metamodels["svm"] = svm
#
#         meta_preds = []
#         # If calibrator is not None, fit
#         if self.metamodel_calibrators is not None:
#             if len(np.unique(y_test)) == 1:
#                 if 1 in y_test:
#                     self.return_all_true = True
#                     print(
#                         'The base model has an accuracy of 100 percent on the test set. Return predictions of only 100 percent')
#                     self.fit_status = True
#                     return
#                 else:
#                     self.return_all_false = True
#                     print(
#                         'The base model has an accuracy of 0 percent on the test set. Return predictions of only 0 percent')
#                     self.fit_status = True
#                     return
#
#             logger.info("Metamodels considered %s", self.metamodels_considered)
#             for mm in self.metamodels_considered:
#                 model = self.metamodels[mm]
#                 preds = model.predict_proba(x_test)
#
#                 meta_preds.append(preds)
#                 preds = preds[:, 1]
#                 self.metamodel_calibrators[mm].fit(preds, y_test)
#
#         meta_preds = np.asarray(meta_preds)
#         meta_preds = np.mean(meta_preds, axis=0)
#         meta_preds = meta_preds[:, 1]
#
#         self.calibrator.fit(meta_preds, y_test)
#
#         self.fit_status = True
#
#     def predict(self, X_unprocessed, X):
#         X = X.values
#         assert self.fit_status
#         if self.return_all_true:
#             preds = 0.99999999 * np.ones(X.shape[0])
#             output = {'confidences': preds, 'uncertainties': np.zeros(preds.shape)}
#             return output
#
#         if self.return_all_false:
#             preds = 0 * np.ones(X.shape[0])
#             output = {'confidences': preds, 'uncertainties': np.zeros(preds.shape)}
#             return output
#
#         confidences = []
#         for mm in self.metamodels_considered:
#             logger.info("Predicting against metamodel %s", mm)
#             model = self.metamodels[mm]
#             preds = model.predict_proba(X)
#             preds = preds[:, 1]
#
#             if self.metamodel_calibrators:
#                 out = self.metamodel_calibrators[mm].predict(preds)
#                 confidences.append(out)
#             else:
#                 confidences.append(preds)
#
#         preds = confidences
#         confidences = np.mean(preds, axis=1)
#
#         if self.calibrator is not None:
#             confidences = self.calibrator.predict(confidences)
#
#         output = {'confidences': confidences, 'uncertainties': np.zeros(confidences.shape)}
#         return output
#
#     def save(self, output_location):
#         pass
#
#     def load(self, input_location):
#         pass
