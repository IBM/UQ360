
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor
from uq360.utils.gbm_whitebox_features import GBM_WhiteboxFeatureExtractor
from uq360.utils.hpo_search import CustomRandomSearch

"""
Performance predictor for structured data. It is based on a pair of meta-models, one logistic regression and one GBM. 
This performance predictor does not have a method to quantify its own uncertainty, so the uncertainty 
values are zero.  
"""
class StructuredDataPredictor(PerfPredictor):
    def __init__(self, calibrator=None):
        self.metamodel_list = []
        self.return_all_true = False
        self.x_test = None
        self.y_test = None
        super(StructuredDataPredictor, self).__init__(calibrator)

    @classmethod
    def name(cls):
        return ('structured_data')

    def fit(self, x_test_unprocessed, x_test, y_test):
        # stash the unmodified x_test for later
        self.x_test = x_test
        self.y_test = y_test

        x_test = x_test.values
        # Don't split off test set for calibration if calibrator = None
        if self.calibrator is not None:
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
                raise Exception("Cannot train a meta-model on a base model with 0 percent accuracy. Exiting. ")

        # Balance datasets
        x_dev, y_dev = self._balance_data(x_dev, y_dev)

        lr_parameters = {
            "C": np.logspace(-3, 3, 7),
            "penalty": ["l1", "l2"]
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

        # print('fitting gbm meta-model')
        classifier1 = GradientBoostingClassifier()
        clf1 = CustomRandomSearch(classifier1, gbm_parameters, **randomized_params)
        clf1.fit(x_dev, y_dev)

        # print('fitting logistic meta-model')
        classifier2 = LogisticRegression(solver='liblinear', max_iter=200)
        clf2 = CustomRandomSearch(classifier2, lr_parameters, **randomized_params)
        clf2.fit(x_dev, y_dev)

        gbm = clf1.best_estimator_
        self.metamodel_list.append(gbm)
        self.metamodel_list.append(clf2.best_estimator_)
        self.classifier_for_tree_whitebox_features = gbm

        # If calibrator is not None, fit
        if self.calibrator is not None:
            if len(np.unique(y_test)) == 1:
                if 1 in y_test:
                    self.return_all_true = True
                    print(
                        'The base model has an accuracy of 100 percent on the test set. Return predictions of only 100 percent')
                    self.fit_status = True
                    return
                else:
                    raise Exception("Cannot train a calibrator on a base model with 0 percent accuracy. Exiting. ")

            meta_preds = []
            for mm in self.metamodel_list:
                preds = mm.predict_proba(x_test)
                meta_preds.append(preds)

            meta_preds = np.asarray(meta_preds)
            meta_preds = np.mean(meta_preds, axis=0)
            meta_preds = meta_preds[:, 1]
            self.calibrator.fit(meta_preds, y_test)
        self.fit_status = True

    def predict(self, X_unprocessed, X):
        X = X.values
        # TODO: add some sanity checks for X
        assert self.fit_status

        # add all known whitebox features here with value 0, so they aren't missing if jump out of the predictor early
        self.init_all_whitebox_features()

        if self.return_all_true:
            preds = 0.99999999 * np.ones(X.shape[0])
            output = {'confidences': preds, 'uncertainties': np.zeros(preds.shape)}
            return output

        preds = []
        for mm in self.metamodel_list:
            preds.append(mm.predict_proba(X))

        # Just check the first element
        try:
            assert (len(preds[0].shape) == 2)  # ... and we are assuming the class indexed by 1 is "success/correct"
        except AssertionError:
            raise Exception("Metamodel probabilities have incorrect shape {}".format(preds[0].shape))

        # Make preds an (N_samples, 2) array.
        preds = np.asarray(preds)[:, :, 1]
        preds = np.transpose(preds)
        # Average predictions from gbm and logistic regression metamodels
        confidences = np.mean(preds, axis=1)

        # Create some white-box features
        # I want to use "delta between the two models in the ensemble", but to make it work for ensembles > 2, I'm using stdev instead of delta
        # stdev of 2 things is the same as 0.5 * the delta, so it should be identical as a feature

        # stdev_of_means: computes the mean for each model first, then takes the stdev (aka delta) of them
        stdev_of_means = np.std(np.mean(preds, axis=0))
        self.add_whitebox_feature('stdev_of_means', stdev_of_means)

        # mean_of_stdevs: computes the stdev for each prediction first, which is analogous to the abs(delta) for each prediction
        # then takes mean of them.  So this measure how often the predictions vary, even if the means turned out to be the same (because
        # the deltas canceled out)
        mean_of_stdevs = np.mean(np.std(preds, axis=1))
        self.add_whitebox_feature('mean_of_stdevs', mean_of_stdevs)

        pre_calibration_accuracy = np.mean(confidences)

        if self.calibrator is not None:
            confidences = self.calibrator.predict(confidences)

        post_calibration_accuracy = np.mean(confidences)
        calibration_impact = post_calibration_accuracy - pre_calibration_accuracy
        self.add_whitebox_feature('calibration_impact', calibration_impact)
        self.add_whitebox_feature('calibration_impact_abs', abs(calibration_impact))

        gbm_feat_extractor = GBM_WhiteboxFeatureExtractor(self.classifier_for_tree_whitebox_features)
        gbm_features = gbm_feat_extractor.compute_gbm_internal_whitebox_features(self.x_test.to_numpy(), self.y_test, X)

        # Copy gbm features into predictor's whitebox features
        # TODO:  this copying should be abstracted.  But not doing it before this round of runs
        for k, v in gbm_features.items():
            self.add_whitebox_feature(k, v)

        output = {'confidences': confidences, 'uncertainties': np.zeros(confidences.shape)}
        return output

    # Add all whitebox features
    # This should not be hardcoded here but more of a transformer architecture
    # But we need this asap so we can reeneable assertions (these keep tripping them)
    def init_all_whitebox_features(self):
        self.init_whitebox_feature('calibration_impact')
        self.init_whitebox_feature('calibration_impact_abs')
        self.init_whitebox_feature('stdev_of_means')
        self.init_whitebox_feature('mean_of_stdevs')
        self.init_whitebox_feature('gbm_node_freq_delta_abs_max')
        self.init_whitebox_feature('gbm_node_freq_delta_abs_sum')
        self.init_whitebox_feature('gbm_node_freq_delta_abs_std')
        self.init_whitebox_feature('gbm_node_accuracy_delta')
        self.init_whitebox_feature('gbm_node_accuracy_delta_abs')
        for i in range(0, 4):
            self.init_whitebox_feature('gbm_delta_depth_' + str(i))
            self.init_whitebox_feature('gbm_delta_depth_abs_' + str(i))
        self.init_whitebox_feature('gbm_delta_sum')
        self.init_whitebox_feature('gbm_delta_sum_abs')

    def save(self, output_location):
        self.register_pkl_object(self.metamodel_list[0], 'gbm-metamodel')
        self.register_pkl_object(self.metamodel_list[1], 'logistic-metamodel')

        output_json = {}
        if self.return_all_true:
            output_json['return_all_true'] = 'true'
        else:
            output_json['return_all_true'] = 'false'
        self.register_json_object(output_json, 'output_dict')
        self._save(output_location)

    def load(self, input_location):
        self._load(input_location)

        self.metamodel_list = [None] * 2
        pkl_objs, pkl_names = self.pkl_registry
        for obj, name in zip(pkl_objs, pkl_names):
            if name == 'gbm-metamodel':
                self.metamodel_list[0] = obj
                assert type(self.metamodel_list[0]) == GradientBoostingClassifier
            elif name == 'logistic-metamodel':
                self.metamodel_list[1] = obj
                assert type(self.metamodel_list[1]) == LogisticRegression

        json_objs, json_names = self.json_registry
        for obj, name in zip(json_objs, json_names):
            if name == 'output_dict':
                if obj['return_all_true'] == 'true':
                    self.return_all_true = True
                elif obj['return_all_true'] == 'false':
                    self.return_all_true = False
                else:
                    raise Exception(
                        "Key 'return_all_true' must be either 'true' or 'false'. Cannot load StructuredData Predictor. ")

        self.fit_status = True
        self.transformers_fit_flag = True


