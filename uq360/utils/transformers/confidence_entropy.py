from scipy.stats import entropy
from uq360.utils.transformers.feature_transformer import FeatureTransformer

class ConfidenceEntropyTransformer(FeatureTransformer):
    """Entropy of the confidence vector from the input/base model. """
    def __init__(self):
        super(ConfidenceEntropyTransformer, self).__init__()

    @classmethod
    def name(cls):
        return ('confidence_entropy')

    def transform(self, x, predictions):
        return entropy(predictions.T) # scipy.stats entropy calculates entropy along axis=0, which is the batch dim

    def save(self, output_dir=None):
        pass

    def load(self, input_dir=None):
        pass