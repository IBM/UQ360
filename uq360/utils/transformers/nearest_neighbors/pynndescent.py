from typing import Union, List

try:
    import pynndescent
except ImportError as e:
    raise ImportError(
        f"This optional module depends on the uninstalled 'pynndescent' optional dependency: {__name__}"
    ) from e

from .base import BaseNearestNeighbors


class PyNNDNearestNeighbors(BaseNearestNeighbors):
    @classmethod
    def name(cls):
        return "pynndescent_nearest_neighbors"

    def __init__(self):
        super(PyNNDNearestNeighbors, self).__init__()
        self.index = None

    def fit(self, X, **kwargs):
        self.index = pynndescent.NNDescent(X, **kwargs)
        self.index.prepare()

        return self

    def transform(self, X, n_neighbors):
        indices, distances = self.index.query(X, n_neighbors)

        return distances, indices

    def save(self, output_location=None):
        self.register_pkl_object(self.index, "index")
        self._save(output_location=output_location)

    def load(self, input_location=None):
        self._load(input_location=input_location)
        self.index = self.pkl_registry[0][0]
