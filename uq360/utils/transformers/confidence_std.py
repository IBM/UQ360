
import numpy as np
from uq360.utils.transformers.feature_transformer import FeatureTransformer

class ConfidenceStdTransformer(FeatureTransformer):
    """Standard deviation of the confidence vector from the input/base model. """
    def __init__(self):
        super(ConfidenceStdTransformer, self).__init__()
    
    @classmethod
    def name(cls):
        return ('confidence_std')

    def transform(self, x, predictions):
        return np.std(predictions, axis=1)

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass
