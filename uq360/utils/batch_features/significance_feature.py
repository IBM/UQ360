
import numpy as np
from uq360.utils.batch_features.batch_feature import BatchFeature
from uq360.utils.significance_test import SignificanceTester
from uq360.utils.transformers.confidence_top import ConfidenceTopTransformer


class SignificanceFeature(BatchFeature):
    """
    Batch feature which uses bootstrap to compute 95% confidence intervals for the average value
    of the highest class confidence over all points in the production set at inference time.

    The output is the width of this confidence interval.
    """
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
