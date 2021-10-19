import traceback
import numpy as np
from uq360.utils.batch_features.blackbox_feature import BlackboxFeature

from uq360.utils.batch_features.feature_extractor import FeatureExtractor
from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor
from uq360.utils.batch_features.drift_classifier import DriftClassifier

from uq360.utils.utils import Timer


# This class is the new wrapper class that will drive all aspects of PP and Error Bars
class PredictorDriver(object):

    def __init__(self, pp_type,
                 base_model=None,
                 pointwise_features=None,
                 batch_features=None,
                 blackbox_features=None,
                 use_whitebox=True,
                 use_drift_classifier=True,
                 uncertainty_model_file=None,
                 uncertainty_model_type=None,
                 calibrator='isotonic_regression'):
        self.timer = Timer()
        self.timer.start('init')
        self.mode = 'prod'
        # Specify the performance predictor that you would like to run
        self.perf_predictor = PerfPredictor.instance(pp_type, calibrator=calibrator)

        self.pointwise_features = None
        self.batch_features = None
        self.blackbox_features = None

        if pointwise_features is not None:
            self.pointwise_features = pointwise_features
        else:
            pointwise_features_common = ["confidence_top", "confidence_delta", "confidence_entropy", "gbm",
                                          "class_frequency"]
            if pp_type in ['text_ensemble', 'text_ensemble_v1', 'text_ensemble_v2']:
                self.pointwise_features = pointwise_features_common + ["mlp", "svc"]
            elif pp_type in ['passthrough']:
                self.pointwise_features = ["confidence_top"]
            else:

                self.pointwise_features = pointwise_features_common + ["random_forest",
                                                                       "logistic_regression", "one_class_svm",
                                                                       "umap_kde"]

        if batch_features is not None:
            self.batch_features = batch_features
        else:
            if pp_type not in ['text_ensemble', 'passthrough', 'text_ensemble_v1', 'text_ensemble_v2']:
                self.batch_features = ["confidence_top_distance", "confidence_delta_distance",
                                       "confidence_entropy_distance",
                                       "random_forest_distance", "gbm_distance", "logistic_regression_distance",
                                       "pca_distance", "best_feature_distance",
                                       "class_frequency_distance", "num_important_features", "bootstrap"]

        if blackbox_features is not None:
            self.blackbox_features = blackbox_features
        else:
            if pp_type not in ['text_ensemble', 'passthrough', 'text_ensemble_v1',  'text_ensemble_v2']:
                self.blackbox_features = ["pp_std", "pp_entropy", "pp_bootstrap", "base_model_entropy_ratio",
                                          "predicted_accuracy_change", "pp_uncertainty"]

        self.blackbox_payloads = {}
        self.use_whitebox = use_whitebox
        self.use_drift_classifier = use_drift_classifier
        self.base_model = base_model
        print("Batch features :", self.batch_features)
        print("Pointwise features :", self.pointwise_features)
        print("Blackbox features :", self.blackbox_features)
        print("Predictor type :", pp_type)
        self.feature_extractor = FeatureExtractor(self.pointwise_features, self.batch_features)

        # TODO: have a flag to turn this off?
        if uncertainty_model_type is None:
            self.um = None
        # else:
        #     assert uncertainty_model_file is not None
        #     assert os.path.isfile(uncertainty_model_file)
        #     self.um = UncertaintyModel.instance(uncertainty_model_type)
        #     self.um.load_from_file(uncertainty_model_file, calibrator_file=None, ucc_formula=None)

        self.timer.stop('init')

    def fit(self, x_train, y_train, x_test, y_test, test_predicted_probabilities=None):
        self.timer.start('fit')
        self.train_labels = np.unique(y_train)
        # Fit the feature extractors
        self.feature_extractor.fit(x_train, y_train)

        # Get metamodel ground truth
        if self.base_model is not None:
            test_predicted_probabilities = self.base_model.predict_proba(x_test)
            predictions = self.base_model.predict(x_test)
        else:
            try:
                assert test_predicted_probabilities is not None
            except:
                raise Exception(
                    "If base model is not provided to constructor, confidence vectors must be passed to 'fit'")
            predictions_unconverted = np.argmax(test_predicted_probabilities, axis=1)
            predictions = np.array([self.train_labels[x] for x in predictions_unconverted])

        # Collect the point wise features for test
        test_features, self.payloads = self.feature_extractor.transform_test(x_test,
                                                                             predicted_probabilities=test_predicted_probabilities)

        # Now invoke the performance predictor
        y_meta = np.where(predictions == np.squeeze(y_test), 1, 0)
        self.perf_predictor.fit(x_test, test_features, y_meta)

        if self.blackbox_features:
            for bb in self.blackbox_features:
                self.blackbox_payloads[bb] = BlackboxFeature.instance(bb,
                                                                      train_labels=self.train_labels).extract_payload(
                    test_predicted_probabilities, y_test)

        self.x_test = x_test
        self.x_test_features = test_features

        self.timer.stop('fit')

    def predict(self, x_prod, predicted_probabilities=None):
        self.timer.start('predict')
        result = {}

        if self.base_model is not None:
            predicted_probabilities = self.base_model.predict_proba(x_prod)
        else:
            try:
                assert predicted_probabilities is not None
            except:
                raise Exception(
                    "If base model is not provided to constructor, confidence vectors must be passed to 'predict'")

        # Collect the pointwise features for x_prod
        prod_features, batch_features = self.feature_extractor.transform_prod(x_prod, predicted_probabilities,
                                                                              self.payloads)

        # Get the PP predictions on prod
        predictor_output = self.perf_predictor.predict(x_prod, prod_features)

        accuracy = 100.0 * np.mean(predictor_output['confidences'])
        uncertainty = 100.0 * np.mean(predictor_output['uncertainties'])

        # Predict with uncertainty model
        if self.blackbox_features:
            for bb in self.blackbox_features:
                feature = BlackboxFeature.instance(bb, train_labels=self.train_labels)
                batch_features[bb] = feature.extract_blackbox_feature(predicted_probabilities, predictor_output,
                                                                      self.blackbox_payloads[bb])

        if self.use_whitebox:
            whitebox = self.perf_predictor.whitebox_features
            for wb in whitebox.keys():
                if wb in batch_features:
                    raise Exception(
                        "{} is already a feature in 'batch_features'. Cannot reuse as a whitebox feature. ".format(wb))
                batch_features[wb] = whitebox[wb]

        if self.use_drift_classifier:
            x_test = self.x_test
            x_test_features = self.x_test_features

            test_combined = np.column_stack([x_test, x_test_features])
            prod_combined = np.column_stack([x_prod, prod_features])

            dc_names = ['orig', 'computed', 'combined']
            test_sets = [x_test, x_test_features, test_combined]
            prod_sets = [x_prod, prod_features, prod_combined]
            for i in range(len(dc_names)):
                dc = DriftClassifier(dc_names[i])
                acc, dist = dc.fit_predict(test_sets[i], prod_sets[i])
                label = 'drift_classifier_' + dc.name + '_'
                batch_features[label + 'accuracy'] = acc
                for d in dist:
                    batch_features[label + d[0]] = d[1]

        if self.um is not None:
            try:

                bf = batch_features.values
                accuracy, uncertainty = self.um.predict(bf, accuracy)
            except Exception as e:
                if hasattr(self, 'mode') and self.mode == 'test':
                    traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                    print(traceback_str)
                    print('Error occurred while using the uncertainty model', e)
                    result['error'] = str(e)
                else:
                    raise Exception(e)

        result['pointwise_confidences'] = predictor_output['confidences']
        result['pointwise_uncertainties'] = predictor_output['uncertainties']
        # Convert to percent
        result['accuracy'] = accuracy
        result['uncertainty'] = uncertainty
        result['batch_features'] = batch_features

        self.timer.stop('predict')
        return result