
import numpy as np

from uq360.utils.batch_features.batch_feature import BatchFeature
from uq360.utils.batch_features.histogram_utilities import compute_hellinger
from uq360.utils.transformers.distribution_clustering import DistributionClusteringTransformer


# Base class for all clustering based features
class ClusteringFeature(BatchFeature):
    def __init__(self):
        super().__init__()

    # Construct a single histogram
    def extract_histogram(self, vec):
        hist , _ = np.histogram(vec, bins=self.bins, range=(0.0, 1.0))
        hist = np.divide(hist, float(len(vec)))
        return hist

    # Extract the feature vector from the data
    def extract_vector(self, x, predictions):
        vec = self.pointwise_transformer.transform(x, predictions)
        return vec

    # Compute the distance
    def compute_distance(self, hist1, hist2):
        distance = compute_hellinger(hist1, hist2)
        return distance
    
    # Extract features and create histograms
    def extract_features(self, x, predictions):
        vec = self.extract_vector(x, predictions)
        assert(len(vec.shape) == 1)
        
        histogram = self.extract_histogram(vec)
        return vec, histogram

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        vec, histogram = self.extract_features(x, predictions)
        payload = {"histogram": histogram.tolist()}
        return vec, payload
    
    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):

        vec, histogram = self.extract_features(x, predictions)
        distance = self.compute_distance(histogram, np.array(payload['histogram']))
        return vec, distance


"""Batchwise clustering feature where the clustering is performed using Wasserstein distance. """
class WassersteinClustersFeature(ClusteringFeature):
    def __init__(self, scaling_exponent=6, min_cluster_points=12):
        super().__init__()
        tr = DistributionClusteringTransformer(scaling_exponent=scaling_exponent, min_cluster_points=min_cluster_points)
        self.set_transformer('distribution_clustering', tr)
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