
import numpy as np

from uq360.utils.batch_features.histogram_feature import SingleHistogramFeature
from uq360.utils.transformers.confidence_delta import ConfidenceDeltaTransformer
from uq360.utils.transformers.confidence_top import ConfidenceTopTransformer
from uq360.utils.transformers.confidence_entropy import ConfidenceEntropyTransformer
from uq360.utils.transformers.class_frequency import ClassFrequencyTransformer


class BasicPointwiseHistogramDistance(SingleHistogramFeature):
    def __init__(self, bins=10):
        super().__init__(bins)
        self.fit_status = True

    def set_pointwise_transformer(self, pointwise_transformer):
        pass

    
# Top Confidence feature
class BatchConfidenceTop(BasicPointwiseHistogramDistance):
    def __init__(self):
        super().__init__()
        self.set_transformer('confidence_top', ConfidenceTopTransformer())

    @classmethod
    def name(cls):
        return ('confidence_top_distance')

    # Construct a single histogram
    def extract_histogram(self, vec):
        bins = np.concatenate([np.linspace(0,0.9,num=10), np.linspace(0.91,1.0,num=10)])
        self.histogram_edges = bins
        hist , _ = np.histogram(vec, bins=bins)
        hist = np.divide(hist, float(len(vec)))
        return hist


# Confidence Delta feature
class BatchConfidenceDelta(BasicPointwiseHistogramDistance):
    def __init__(self):
        super().__init__()
        self.set_transformer('confidence_delta', ConfidenceDeltaTransformer())

    @classmethod
    def name(cls):
        return ('confidence_delta_distance')


# Confidence Entropy feature
class BatchConfidenceEntropy(BasicPointwiseHistogramDistance):
    def __init__(self):
        super().__init__()
        self.set_transformer('confidence_entropy', ConfidenceEntropyTransformer())
        self.changed_histogram = None

    @classmethod
    def name(cls):
        return ('confidence_entropy_distance')

    # Construct a single histogram
    def extract_histogram(self, vec):
        epsilon = 0.001
        bins = np.concatenate([np.linspace(0,0.1,num=11), np.linspace(0.2,3.0,num=29)])
        # Safety check in case your histogram misses. 
        too_high = np.mean([vec >= max(bins)])
        too_low = np.mean([vec <= min(bins)])
        if too_high > 0.5 or too_low > 0.5:
            if self.changed_histogram != 'false':
                # Don't change prod if test wasn't changed
                bins = np.linspace(min(vec) - epsilon, max(vec)+epsilon, num=25)
                print("Fixing too high, new histogram is ", bins)
        else:
            self.changed_histogram = 'false'
        self.histogram_edges = bins
        hist , _ = np.histogram(vec, bins=bins)
        hist = np.divide(hist, float(len(vec)))
        return hist


# Predicted class frequency ratio
class BatchClassFrequency(BasicPointwiseHistogramDistance):
    def __init__(self):
        super().__init__()
        self.set_transformer('class_frequency', ClassFrequencyTransformer())
        self.fit_status = False

    @classmethod
    def name(cls):
        return ('class_frequency_distance')

    def fit(self, x, y):
        if self.fit_status:
            return
        else:
            self.pointwise_transformer.fit(x,y)
            self.fit_status = True

    # Construct a single histogram
    def extract_histogram(self, vec):
        freq = self.pointwise_transformer.class_frequencies
        ordered_freq = sorted(freq)

        # Left edge, edges between each pair of frequencies, and right edge
        bins = [ordered_freq[0] - 1]
        lf = len(freq)-1
        for i in range(lf):
            bins.append(0.5*(ordered_freq[i]+ordered_freq[i+1]))
        bins.append(ordered_freq[-1] + 1)
        self.histogram_edges = bins
        hist , _ = np.histogram(vec, bins=bins, density=False)
        hist = np.divide(hist, float(len(vec)))
        return hist
