from sklearn.svm import OneClassSVM
from uq360.utils.transformers.feature_transformer import FeatureTransformer


class OneClassSVMTransformer(FeatureTransformer):
    """One-class SVM outlier-classifier based derived feature.
    This transformer fits an SVM decision boundary enclosing the
    full training set. This is then the decision boundary to identify
    outliers in production data at inference time. """
    def __init__(self):
        super(OneClassSVMTransformer, self).__init__()
        self.one_class_classifier = OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('one_class_svm')

    def fit(self, x, y):
        self.one_class_classifier.fit(x)
        self.fit_status = True

    def transform(self, x, predictions):
        return self.one_class_classifier.decision_function(x)

    def save(self, output_location=None):
        self.register_pkl_object(self.one_class_classifier, 'one_class_classifier')
        super(OneClassSVMTransformer, self)._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        self.one_class_classifier = self.pkl_registry[0][0]
        assert type(self.one_class_classifier) == OneClassSVM
        self.fit_status = True
