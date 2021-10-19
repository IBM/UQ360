
import numpy as np
from sklearn.linear_model import LogisticRegression
from uq360.utils.transformers.feature_transformer import FeatureTransformer
from uq360.utils.transformers.confidence_top import ConfidenceTopTransformer
from uq360.utils.transformers.confidence_delta import ConfidenceDeltaTransformer


class LogisticRegressionTransformer(FeatureTransformer):
    """Logistic regression shadow-model feature. This class trains a GBM model on the same train set as
    the input/baseline model. At inference time, the top class confidence and top - 2nd class
    confidence are used as the derived feature. """
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
