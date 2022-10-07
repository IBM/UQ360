import numpy as np

from uq360.utils.nearest_neighbors import BaseNearestNeighbors


class Knn:
    def __init__(
        self,
        nearest_neighbors: BaseNearestNeighbors,
        nearest_neighbors_kwargs={},
    ):
        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbors_kwargs = nearest_neighbors_kwargs

    def fit(self, X):
        self.index = self.nearest_neighbors().fit(X, **self.nearest_neighbors_kwargs)

        return self

    def score(self, X, k, method="knn"):

        dist, idxs = self.index.kneighbors(X, k)

        if method == "knn":
            return np.max(dist, axis=1)
        elif method == "avg":
            return np.mean(dist, axis=1)
        elif method == "lid":
            return -k / np.sum(np.log(dist / np.max(dist)))
        elif method is None:
            return dist, idxs
        else:
            raise ValueError()
