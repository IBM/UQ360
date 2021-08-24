
import numpy as np
from uq360.batch_features.batch_feature import BatchFeature
from uq360.transformers.random_forest import RandomForestTransformer
from uq360.transformers.gbm import GBMTransformer


class BatchNumImportant(BatchFeature):
    def __init__(self, importance_model='random_forest'):
        assert importance_model in ['random_forest', 'gbm']
        if importance_model == 'random_forest':
            tr = RandomForestTransformer()
        else:
            tr = GBMTransformer()
        self.set_transformer(importance_model, tr)
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('num_important_features')

    def set_pointwise_transformer(self, pointwise_transformer):
        self.pointwise_transformer = pointwise_transformer
        if pointwise_transformer.fit_status:
            self.set_importances()
            self.fit_status = True

    def set_importances(self):
        self.importances = self.pointwise_transformer.model.feature_importances_
        ordered_importances = sorted(self.importances)
        importance = 0.0
        i = 0
        while importance < 0.9:
            importance += ordered_importances[i]
            i+=1
        self.num_important = i

    def fit(self, x, y):
        if self.fit_status:
            return
        else:
            self.pointwise_transformer.fit(x,y)
            self.set_importances()
            self.fit_status = True

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        assert self.fit_status
        vec = np.array([])
        payload = {"num_important": self.num_important}
        return vec, payload
    
    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        vec = np.array([])
        num_important = payload['num_important']
        return vec, num_important