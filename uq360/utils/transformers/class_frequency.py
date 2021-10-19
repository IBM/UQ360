
import numpy as np
from uq360.utils.transformers.feature_transformer import FeatureTransformer

class ClassFrequencyTransformer(FeatureTransformer):
    """Fraction of the train set belonging to the predicted class. """
    def __init__(self):
        super(ClassFrequencyTransformer, self).__init__()

    @classmethod
    def name(cls):
        return ('class_frequency')

    def fit(self, x, y):
        _, class_counts = np.unique(y, return_counts=True)
        self.class_frequencies = [float(c) /float(len(y)) for c in class_counts]
        self.fit_status = True

    def transform(self, x, predictions):
        pred_classes = list(np.argmax(predictions, axis=1))
        class_frequencies = [self.class_frequencies[x] for x in pred_classes]
        return np.array(class_frequencies)

    def save(self, output_location=None):
        json_dump = {"frequencies": self.class_frequencies}
        self.register_json_object(json_dump, 'class_frequencies')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        self.class_frequencies = self.json_registry[0][0]['frequencies']
        assert type(self.class_frequencies) == list
        self.fit_status = True
