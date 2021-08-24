
import numpy as np

from uq360.batch_features.histogram_feature import MultiHistogramFeature
from uq360.transformers.gbm import GBMTransformer
from uq360.transformers.logistic_regression import LogisticRegressionTransformer
from uq360.transformers.random_forest import RandomForestTransformer


class BatchShadowModel(MultiHistogramFeature):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.fit_status = False



    def set_pointwise_transformer(self, pointwise_transformer):
        self.pointwise_transformer = pointwise_transformer
        if pointwise_transformer.fit_status:
            self.fit_status = True

    def fit(self, x, y):
        if self.fit_status:
            return
        else:
            self.pointwise_transformer.fit(x,y)
            self.fit_status = True





class BatchShadowGBM(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('gbm', GBMTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('gbm_distance')




class BatchShadowLogisticRegression(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('logistic_regression', LogisticRegressionTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('logistic_regression_distance')


class BatchShadowRandomForest(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('random_forest', RandomForestTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('random_forest_distance')

