from collections import OrderedDict

import numpy as np
import pandas as pd

from uq360.utils.batch_features.batch_feature import BatchFeature
from uq360.utils.transformers import FeatureTransformer

class FeatureExtractor:
    '''
    Class to perform feature extraction. Pointwise features have per-datapoint values, and can be computed for any
    samples. Batch features are computed for paired datasets (ie a test set and an unlabeled production set). The
    batch features often take the pointwise feature values as input, for example to construct dataset level histograms.
    '''
    def __init__(self, pointwise_features, batch_features):
        self.fit_flag = False

        self.pointwise_features = []
        self.pointwise_feature_objects = {}

        self.batch_features = []
        self.batch_feature_objects = {}

        for pwf in pointwise_features:
            if type(pwf) == str:
                tr = FeatureTransformer.instance(pwf)
                self.pointwise_features.append(pwf)
                self.pointwise_feature_objects[pwf] = tr
            elif type(pwf) == list:
                assert len(pwf) == 2
                assert type(pwf[0]) == str
                assert type(pwf[1]) == dict
                tr = FeatureTransformer.instance(pwf[0], **pwf[1])
                self.pointwise_features.append(pwf[0])
                self.pointwise_feature_objects[pwf[0]] = tr
        if batch_features:
            for bf in batch_features:
                tr = BatchFeature.instance(bf)
                self.batch_features.append(bf)
                self.batch_feature_objects[bf] = tr

    def fit(self, x: np.ndarray, y: np.ndarray):
        for pwfo in self.pointwise_feature_objects.values():
            pwfo.fit(x, y)

        for bfo in self.batch_feature_objects.values():
            ptype = bfo.pointwise_type
            # If the required pointwise transformer already exists, point the batch feature to it
            if ptype in self.pointwise_features:
                bfo.set_pointwise_transformer(self.pointwise_feature_objects[ptype])
            # Fit should check if we already loaded from a pointwise transformer
            bfo.fit(x, y)
        self.fit_flag = True

    # Logic shared across transforming test and prod
    def _get_features(self, x, predicted_probabilities):
        assert self.fit_flag
        features = OrderedDict()
        for pwf, pwfo in self.pointwise_feature_objects.items():
            feature = pwfo.transform(x, predicted_probabilities)
            feature = np.squeeze(feature)
            if len(feature.shape) > 1:
                lth = feature.shape[1]
                for l in range(lth):
                    features[pwf+'_'+str(l+1)] = feature[:,l]

            else:
                features[pwf] = feature
        print('Features extracted for :', features.keys())
        features = pd.DataFrame(features, columns=features.keys())
        return features

    def transform_test(self, x: np.ndarray, predicted_probabilities: np.ndarray):

        features = self._get_features(x, predicted_probabilities)

        payloads = {}
        for bf, bfo in self.batch_feature_objects.items():
            _, payload = bfo.extract_pointwise_and_payload(x, predicted_probabilities)
            payloads[bf] = payload

        return features, payloads

    def transform_prod(self, x: np.ndarray, predicted_probabilities: np.ndarray, payloads: dict):
        features = self._get_features(x, predicted_probabilities)

        batch_features = OrderedDict()
        for bf, bfo in self.batch_feature_objects.items():
            _, distance = bfo.extract_pointwise_and_batch(x, predicted_probabilities, payloads[bf])
            if type(distance) in [float, int]:
                # If distance is a number, us it with the name of the feature class's name
                batch_features[bf] = distance
            elif type(distance) == list:
                # If distance is a list, use the featue class's name with various suffixes
                i = -1
                for d in distance:
                    i += 1
                    if type(d) == tuple:
                        # If a tuple is returned, use the first element of the tuple for the suffix
                        feature_name = bf + '_' + d[0]
                        distance = d[1]
                    else:
                        # If the list is just numbers, assign integer counts
                        # This is a legacy to support old features we havent' converted yet
                        feature_name = bf + '_' + str(i)
                        distance = d
                    batch_features[feature_name] = distance

            else:
                raise Exception("Unknown type for batch feature: {}".format(type(distance)))

        batch_features = pd.Series(batch_features)
        return features, batch_features
