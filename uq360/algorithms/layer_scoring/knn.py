import numpy as np

from uq360.algorithms.layer_scoring.latent_scorer import LatentScorer
from uq360.utils.transformers.nearest_neighbors import BaseNearestNeighbors


class KNNScorer(LatentScorer):
    """KNN-based latent space anomaly detector. Return some measure of distance to the training data."""

    def _process_pretrained_model(self, *argv, **kwargs):
        pass

    def get_params(self):
        return {
            'nearest_neighbors': self.nearest_neighbors.__class__.__name__,
            'nearest_neighbor_kwargs': self.nearest_neighbors_kwargs,
        }

    def __init__(
            self,
            n_neighbors: int,
            method: str = "knn",
            nearest_neighbors: BaseNearestNeighbors = None,
            nearest_neighbors_kwargs={},
            model=None,
            layer=None
    ):
        """
        
        Args:
            n_neighbors: number of nearest neighbors to consider in in-distribution data
            method: one of ("knn", "avg", "lid"). These correspond respectively to the distance to the k-th neighbor, the mean of the kNN,
            nearest_neighbors: nearest neighbor algorithm, see uq360.utils.transformers.nearest_neighbors
            nearest_neighbors_kwargs: keyword arguments for the NN algorithm
            model: torch Module to analyze
            layer: layer (torch Module) inside the model whose output is to be analyzed
        Notes:
            The model and layer arguments are optional.
            If no model or layer is provided, it is expected that the inputs are already latent vectors.
            If both a model and layers are provided, inputs are expected to be model inputs
            to be mapped to latent vectors.
        """
        if nearest_neighbors is None:
            raise ValueError(
                "nearest neighbor must be nearest neighbor algorithm. See uq360.utils.transformers.nearest_neighbors")
        super(KNNScorer, self).__init__(model=model, layer=layer)
        self.n_neighbors = n_neighbors
        assert method in ("knn", "avg", "lid")
        self.method = method

        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbors_kwargs = nearest_neighbors_kwargs
        self.index = None

    def fit(self, X: np.ndarray):
        """Register X as in-distribution data"""
        return super(KNNScorer, self).fit(X)

    def _fit(self, X):
        self.index = self.nearest_neighbors().fit(X, **self.nearest_neighbors_kwargs)

        return self

    def predict(self, X: np.ndarray, n_neighbors=None, method: str = None):
        """Compute a KNN-distance-based anomaly score on query data X.

        Args:
            X: query data
            n_neighbors: number of nearest neighbors to consider in in-distribution data
            method: one of ("knn", "avg", "lid"). These correspond respectively to the distance to the k-th neighbor, the mean of the kNN,

        Returns:
            anomaly scores

        """
        return super(KNNScorer, self).predict(X, n_neighbors, method=method)

    def _predict(self, X, n_neighbors=None, method="knn"):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if method is None:
            method = self.method
        dist, idxs = self.index.transform(X, n_neighbors)

        if method == "knn":
            return np.max(dist, axis=1)
        elif method == "avg":
            return np.mean(dist, axis=1)
        elif method == "lid":
            return -n_neighbors / np.sum(np.log(dist / np.max(dist, axis=1, keepdims=True)), axis=1)
        elif method is None:
            return dist, idxs
        else:
            raise ValueError()
