
import numpy as np
from uq360.base import Base


class BatchFeature(Base):
    '''
    Base class for constructing distribution-level features derived from batches of datapoints
    rather than from single samples.

    These features are engineered to detect data drift between two batches of data, usually a train or test
    set and a batch of unlabeled production data.
    '''
    def __init__(self):
        pass

    # Initial setup for the transformer
    def set_transformer(self, name, transformer):
        self.pointwise_type = name
        self.pointwise_transformer = transformer

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def extract_pointwise_and_payload(self, x: np.ndarray, predictions: np.ndarray):
        pass

    def extract_pointwise_and_batch(self, x: np.ndarray, predictions: np.ndarray, payload: dict):
        pass

    @classmethod
    def instance(cls, subtype_name=None, **params):
        subtype_name = subtype_name

        return super(BatchFeature, cls).instance(subtype_name, **params)

        

