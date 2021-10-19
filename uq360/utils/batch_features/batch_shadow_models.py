from uq360.utils.batch_features.histogram_feature import MultiHistogramFeature
from uq360.utils.transformers.gbm import GBMTransformer
from uq360.utils.transformers.logistic_regression import LogisticRegressionTransformer
from uq360.utils.transformers.random_forest import RandomForestTransformer


class BatchShadowModel(MultiHistogramFeature):
    """
    Base class for using the predictions of shadow-models (models trained on the same data/labels as the input/base model),
    to produce histogram-based batchwise features. The output (highest confidence and 1st - 2nd highest confidence) from
    the shadow model will be histogrammed, and the selected distance metrics will be used to compute the distance between
    the test and production histograms. These distances will be the values of the batch-wise features.
    """
    def __init__(self, bins=10, metrics=None):
        super().__init__(bins, metrics=metrics)
        self.fit_status = False

    def set_pointwise_transformer(self, pointwise_transformer):
        self.pointwise_transformer = pointwise_transformer
        if pointwise_transformer.fit_status:
            self.fit_status = True

    def fit(self, x, y):
        if self.fit_status:
            return
        else:
            self.pointwise_transformer.fit(x, y)
            self.fit_status = True


"""Batch shadow-model where the shadow-model type is a GBM. """
class BatchShadowGBM(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('gbm', GBMTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('gbm_distance')


"""Batch shadow-model where the shadow-model type is a logistic regression model. """
class BatchShadowLogisticRegression(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('logistic_regression', LogisticRegressionTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('logistic_regression_distance')


"""Batch shadow-model where the shadow-model type is a random forest model. """
class BatchShadowRandomForest(BatchShadowModel):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.set_transformer('random_forest', RandomForestTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('random_forest_distance')
