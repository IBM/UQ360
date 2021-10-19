
import numpy as np

from uq360.utils.batch_features import BatchFeature
from uq360.utils.batch_features.histogram_utilities import compute_KS, compute_scaled_up, compute_hellinger, \
    compute_squared, compute_wasserstein, compute_JS, compute_cosine_similarity


class HistogramFeature(BatchFeature):
    '''
    Base class for all histogram-based batch features.
    These features are derived by constructing a histogram of some sample-wise values
    from the batches of data being compared for drift, and using a probability distribution metric to
    compute the "distance" between the two histograms.

    The shape of the output will depend on how many statistical distances (more accurately statistical divergences) are
    specified. The default is a set of three: 'scale_up', 'scale_up_sq', and 'KS' (Kolmogorov-Smirnov). Other available
    options are: 'hellinger', 'squared', 'JS' (Jensen-Shannon), 'wasserstein', and 'cosine'.

    See batch_features.histogram_utilities for implementations of these distance metrics.
    '''
    def __init__(self, bins, metrics=None):
        super().__init__()
        self.bins = bins
        self.prod_histogram = None
        if metrics is None:
            self.metrics = ['scale_up', 'scale_up_sq', 'KS']  # Default set of distance metrics
        else:
            self.metrics = metrics

        self.available_metrics = ['hellinger', 'squared', 'scale_up', 'scale_up_sq', 'KS', 'JS',
                                  'wasserstein', 'cosine']
        for metric in self.metrics:
            try:
                assert metric in self.available_metrics
            except AssertionError:
                print(f"WARNING: Metric: {metric} not in available metrics: {self.available_metrics}. "
                      f"Skipping this distance metric. ")
                self.metrics.remove(metric)

    # Construct a single histogram
    def extract_histogram(self, vec):
        self.histogram_edges = np.linspace(0.0, 1.0, num=self.bins+1)
        hist , _ = np.histogram(vec, bins=self.histogram_edges)
        hist = np.divide(hist, float(len(vec)))
        return hist

    # Extract the feature vector from the data
    def extract_vector(self, x, predictions):
        vec = self.pointwise_transformer.transform(x, predictions)
        return vec

    # Compute the distance
    def compute_distance(self, hist1, hist2):
        distances = []
        if 'hellinger' in self.metrics:
            hellinger = compute_hellinger(hist1, hist2)
            distances.append(('hellinger', hellinger))

        if 'squared' in self.metrics:
            squared = compute_squared(hist1, hist2)
            distances.append(('squared', squared))

        if 'scale_up' in self.metrics or 'scale_up_sq' in self.metrics:
            scale_up, scale_up_sq = compute_scaled_up(hist1, hist2)
            if 'scale_up' in self.metrics:
                distances.append(('scale_up', scale_up))
            if 'scale_up_sq' in self.metrics:
                distances.append(('scale_up_sq', scale_up_sq))

        if 'wasserstein' in self.metrics:
            bin_centers = [0.5*(self.histogram_edges[i]+self.histogram_edges[i+1]) for i in range(len(self.histogram_edges)-1)]
            wd = compute_wasserstein(bin_centers, bin_centers, 1, prob_A=hist1, prob_B=hist2)
            distances.append(('wasserstein', wd))

        if 'KS' in self.metrics:
            ks = compute_KS(hist1, hist2)
            distances.append(('KS', ks))

        if 'JS' in self.metrics:
            js = compute_JS(hist1, hist2)
            distances.append(('JS', js))

        if 'cosine' in self.metrics:
            cs = compute_cosine_similarity(hist1, hist2)
            distances.append(('cosine', cs))
        return distances


# Histogram Feature that produces only one histogram    
class SingleHistogramFeature(HistogramFeature):
    def __init__(self, bins, metrics=None):
        super().__init__(bins, metrics=metrics)

    # Extract features and create histograms
    def extract_features(self, x, predictions):
        vec = self.extract_vector(x, predictions)
        assert(len(vec.shape) == 1)
        
        histogram = self.extract_histogram(vec)
        return vec, histogram

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        vec, histogram = self.extract_features(x, predictions)
        payload = {"histogram": histogram.tolist(), "bins": self.bins}
        return vec, payload
    
    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        vec, histogram = self.extract_features(x, predictions)
        distances = self.compute_distance(histogram, np.array(payload['histogram']))
        return vec, distances

'''
Features that use transformers that create multiple histograms. These features do not create multi-dimensional 
histograms, they just create a vector of single-dimensional histograms from a vector of input values, and compute
coordinate-wise distances between the single-dimensional histograms. 
'''

class MultiHistogramFeature(HistogramFeature):
    def __init__(self, bins, metrics=None):
        super().__init__(bins, metrics=metrics)

    # Extract features and create histograms
    # Expect the vector returned from the transformer to have >1 columns
    def extract_features(self, x, predictions):
        vec = self.extract_vector(x, predictions)
        assert(len(vec.shape) > 1)
        histograms = []
        num_cols = vec.shape[1]
        for col in range(num_cols):
            hist = self.extract_histogram(vec[:, col])
            histograms.append(hist)
        return vec, histograms

    # Create the name of the histogram in the payload
    def make_payload_key(self, name, i):
        return name + '_' + str(i+1)

    # Extract pointwise features and construct payload to compare to prod set
    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):

        vec, histograms = self.extract_features(x, predictions)

        # Add all the histograms we have into the payload
        payload = {}
        for i in range(len(histograms)):
            h = self.make_payload_key('histogram', i)
            b = self.make_payload_key('bins', i)
            
            payload[h] = histograms[i].tolist()
            payload[b] = self.bins

        return vec, payload

    # Extract pointwise features and compute batch feature from test payload
    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        vec, histograms = self.extract_features(x, predictions)

        # Compute the distances for all the histograms we have and return them as an array
        distances = []
        for i in range(len(histograms)):
            h = self.make_payload_key('histogram', i)
            distance = self.compute_distance(histograms[i], np.array(payload[h]))
            
            for d in distance:
                # Append the index suffex we need to the tuple returned
                d = (d[0] + '_' + str(i), d[1])
                distances.append(d)

        return vec, distances
