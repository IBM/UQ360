
import numpy as np
from uq360.utils.transformers.feature_transformer import FeatureTransformer

class ConfidenceMinMaxTransformer(FeatureTransformer):
    """Ratio of the minimum and maximum class confidences from the input/base model. """
    def __init__(self):
        super(ConfidenceMinMaxTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('confidence_min_max')

    def transform(self, x, predictions):
        return np.min(predictions, axis=1) / np.max(predictions, axis=1)

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass