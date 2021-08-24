
import numpy as np
from uq360.batch_features.batch_feature import BatchFeature
from uq360.utils.significance_test import SignificanceTester
from uq360.transformers.confidence_top import ConfidenceTopTransformer


# TODO: We could do different features based on this same idea
# implement as subclasses
class SignificanceFeature(BatchFeature):
    def __init__(self):
        self.set_transformer('confidence_top', ConfidenceTopTransformer())
        self.fit_status = True

    @classmethod
    def name(cls):
        return ('bootstrap')

    def set_pointwise_transformer(self, pointwise_transformer):
        pass

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        vec = np.array([])
        payload = {}
        return vec, payload

    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        vec = self.pointwise_transformer.transform(x, predictions)
        st = SignificanceTester('average')
        result = st.confidence_interval(vec)
        bootstrap = float(result[2] - result[1])

        return vec, bootstrap
