try:
    import pynndescent
except ImportError as e:
    raise ImportError(
        f"This optional module depends on the uninstalled 'pynndescent' optional dependency: {__name__}"
    ) from e

from .base import BaseNearestNeighbors


class PyNNDNearestNeighbors(BaseNearestNeighbors):
    """Approximate nearest neighbor search using pynndescent.

    Notes
    -----
    This requires the optional depdendency `pynndescent`."""

    @classmethod
    def name(cls):
        return "pynndescent_nearest_neighbors"

    def __init__(self):
        super(PyNNDNearestNeighbors, self).__init__()
        self.index = None

    def fit(self, X, **kwargs):
        """Index a set of reference points

        Args:
            X: a numpy array of reference vectors
            **kwargs: keyword arguments to be passed to pynndescent.NNDescent

        Returns:
            self

        """
        self.index = pynndescent.NNDescent(X, **kwargs)
        self.index.prepare()

        return self

    def transform(self, X, n_neighbors):
        """Perform a k-nearest-neighbor search on a set of query points

        Args:
            X: numpy array of query points
            n_neighbors: number of nearest neighbors to be searched

        Returns:
            a pair of numpy arrays containing respectively the distances to and indices of the k nearest neighbors
            of each query point (each array of shape (X.shape[0], n_neighbors) )
        """
        indices, distances = self.index.query(X, n_neighbors)

        return distances, indices

    def save(self, output_location=None):
        self.register_pkl_object(self.index, "index")
        self._save(output_location=output_location)

    def load(self, input_location=None):
        self._load(input_location=input_location)
        self.index = self.pkl_registry[0][0]
