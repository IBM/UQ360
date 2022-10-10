import numpy as np

from uq360.algorithms.layer_scoring.latent_scorer import LatentScorer
from uq360.utils.transformers.nearest_neighbors import BaseNearestNeighbors


class KNN(LatentScorer):
    def __init__(
            self,
            nearest_neighbors: BaseNearestNeighbors,
            nearest_neighbors_kwargs={},
            model=None,
            layer=None
    ):
        super(KNN, self).__init__(model=model, layer=layer)
        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbors_kwargs = nearest_neighbors_kwargs
        self.index = None

    def _fit(self, X):
        self.index = self.nearest_neighbors().fit(X, **self.nearest_neighbors_kwargs)

        return self

    def _predict(self, X, k, method="knn"):
        dist, idxs = self.index.transform(X, k)

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
