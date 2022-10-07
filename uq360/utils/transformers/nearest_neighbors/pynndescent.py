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
