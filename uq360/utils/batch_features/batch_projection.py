
import numpy as np

from uq360.utils.batch_features.histogram_feature import SingleHistogramFeature
from uq360.utils.batch_features.histogram_utilities import compute_histogram, combine_histograms
from uq360.utils.transformers import PCATransformer
from uq360.utils.transformers.random_forest import RandomForestTransformer
from uq360.utils.transformers.gbm import GBMTransformer


class BatchProjection(SingleHistogramFeature):
    """
           Base class for single-histogram distance based batch-wise features where the values used to construct the histogram
           are a 1-D projection in the original feature space.
    """
    def __init__(self, bins=100):
        super().__init__(bins)
        self.fit_status = False

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        vec, histogram, edges = self.extract_features(x, predictions)
        payload = {"histogram": histogram.tolist(), "histogram_bins": edges.tolist()}
        return vec, payload

    def extract_features(self, x, predictions, quantile=0.9, background_hist=None):
        raise NotImplementedError("All projection based batch features must implement 'extract_features()'")

    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        vec, histogram, edges = self.extract_features(x, predictions, background_hist=np.array(payload['histogram_bins']))
        hist1, hist2, combined_edges = combine_histograms(payload['histogram'], histogram,
                                            payload['histogram_bins'], edges)
        self.histogram_edges = combined_edges
        distance = self.compute_distance(hist1, hist2)
        
        return vec, distance

    # Construct a single histogram
    def extract_histogram(self, vec, quantile, background_hist=None):
        margin = (1.0 - quantile) / 2.0
        quants = np.quantile(vec, [margin, 1.0-margin])
        epsilon = 0.01 * (quants[1]-quants[0])
        lower = quants[0] - epsilon
        upper = quants[1] + epsilon

        truncated_vec = np.array([x for x in vec if x > lower and x < upper])
        if len(truncated_vec) / len(vec) <= 0.5:
            print("SKIPPING VECTOR TRUNCATION")
            truncated_vec = vec
        if background_hist is None:
            hist, edges = np.histogram(truncated_vec, bins=self.bins, range=(lower, upper))
            self.histogram_edges = edges
            hist = np.divide(hist, float(np.sum(hist)))
        else:
            hist, edges = compute_histogram(truncated_vec, bin_number=self.bins, background_histogram=background_hist)
            self.prod_histogram = edges
            hist = np.divide(hist, float(np.sum(hist)))
        return hist, edges


"""Batch-projection feature where the 1-D projection is performed onto the highest PCA component of the data."""
class BatchProjectionPCA(BatchProjection):
    def __init__(self, bins=25):
        super().__init__(bins)
        self.set_transformer('pca', PCATransformer(k=1))
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('pca_distance')

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

    # Extract features and create histograms
    def extract_features(self, x, predictions, quantile=0.9, background_hist=None):
        vec = np.squeeze(self.extract_vector(x, predictions))
        assert(len(vec.shape) == 1)

        histogram, edges = self.extract_histogram(vec, quantile, background_hist=background_hist)
        return vec, histogram, edges


"""Batch-projection feature where the 1-D projection is performed onto feature which a shadow-model 
(default=random forest) considers the highest-importance feature."""
class BatchProjectionHighestImportance(BatchProjection):
    def __init__(self, importance_model='random_forest', bins=25):
        super().__init__(bins)
        assert importance_model in ['random_forest', 'gbm']
        if importance_model == 'random_forest':
            tr = RandomForestTransformer()
        else:
            tr = GBMTransformer()
        self.set_transformer(importance_model, tr)
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('best_feature_distance')

    def fit(self, x, y):
        if self.fit_status:
            return
        else:
            self.pointwise_transformer.fit(x,y)
            self.importances = self.pointwise_transformer.model.feature_importances_
            self.index = np.argmax(self.importances)
            self.fit_status = True

    def set_pointwise_transformer(self, pointwise_transformer):
        self.pointwise_transformer = pointwise_transformer
        if pointwise_transformer.fit_status:
            self.fit_status = True

        try:
            assert self.fit_status
        except:
            raise Exception("Cannot return importances for best feature projection until after fit is performed. ")

        self.importances = self.pointwise_transformer.model.feature_importances_
        self.index = np.argmax(self.importances)

    # Extract features and create histograms
    def extract_features(self, x, predictions, quantile=0.9, background_hist=None):
        vec = x[:,self.index]
        assert(len(vec.shape) == 1)
        histogram, edges = self.extract_histogram(vec, quantile, background_hist=background_hist)
        return vec, histogram, edges
