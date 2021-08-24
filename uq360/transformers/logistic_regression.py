
import numpy as np
from sklearn.linear_model import LogisticRegression
from .feature_transformer import FeatureTransformer
from .confidence_top import ConfidenceTopTransformer
from .confidence_delta import ConfidenceDeltaTransformer


class LogisticRegressionTransformer(FeatureTransformer):
    def __init__(self, model=None):
        super(LogisticRegressionTransformer, self).__init__()

    @classmethod
    def name(cls):
        return ('logistic_regression')

    def fit(self, x, y):
        y = y.ravel()
        self.model = LogisticRegression()
        self.model.fit(x,y)
        self.fit_status = True

    def transform(self, x, predictions):
        assert self.fit_status
        preds = self.model.predict_proba(x)
        top = ConfidenceTopTransformer().transform(x, preds).reshape(-1,1)
        delta = ConfidenceDeltaTransformer().transform(x, preds).reshape(-1,1)
        result = np.concatenate([top, delta], axis=1)
        return result

    def save(self, output_location=None):
        self.register_pkl_object(self.model, 'model')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        self.model = self.pkl_registry[0][0]
        assert type(self.model) == LogisticRegression
        self.fit_status = True
