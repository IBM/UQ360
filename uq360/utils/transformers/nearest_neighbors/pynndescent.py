import pynndescent

from .base import BaseNearestNeighbors


class NearestNeighbors(BaseNearestNeighbors):
    def __init__(self):
        super(NearestNeighbors, self).__init__()
        self.index = None

    def fit(self, X, **kwargs):
        self.index = pynndescent.NNDescent(X, kwargs)
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