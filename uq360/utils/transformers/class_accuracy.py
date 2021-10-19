
import numpy as np
from sklearn.metrics import confusion_matrix

from uq360.utils.transformers.feature_transformer import FeatureTransformer

class ClassAccuracyTransformer(FeatureTransformer):
    """Test set accuracy of the input/baseline model for samples in the predicted class. """

    def __init__(self, model=None):
        if model is None:
            raise Exception("Class accuracy transformer must have a model to initialize. ")
        self.model = model
        super(ClassAccuracyTransformer, self).__init__()

    @classmethod
    def name(cls):
        return ('class_accuracy')

    def fit(self, x, y):
        predictions = self.model.predict(x)
        cm = confusion_matrix(y, predictions)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self.class_accuracies = cm.diagonal()
        self.fit_status = True

    def transform(self, x, predictions):
        pred_classes = list(np.argmax(predictions, axis=1))
        class_accuracies = [self.class_accuracies[x] for x in pred_classes]
        return np.array(class_accuracies)

    def save(self, output_location=None):
        json_dump = {"accuracies": self.class_accuracies}
        self.register_json_object(json_dump, 'class_accuracies')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        self.class_accuracies = self.json_registry[0][0]['accuracies']
        assert type(self.class_accuracies) == list
        self.fit_status = True
