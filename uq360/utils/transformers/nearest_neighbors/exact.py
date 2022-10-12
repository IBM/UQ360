from sklearn.neighbors import NearestNeighbors

from uq360.utils.transformers.nearest_neighbors import BaseNearestNeighbors


class ExactNearestNeighbors(BaseNearestNeighbors):
    """Exact nearest neighbor search using scikit-learn"""

    @classmethod
    def name(cls):
        return "exact_nearest_neighbors"

    def __init__(self):
        super(ExactNearestNeighbors, self).__init__()
        self.index = None

    def fit(self, X, **kwargs):
        """Index a set of reference points

        Args:
            X: a numpy array of reference vectors
            **kwargs: keyword arguments to be passed to sklearn.neighbors.NearestNeighbors

        Returns:
            self

        """
        self.index = NearestNeighbors(**kwargs)
        self.index.fit(X)
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
        return self.index.kneighbors(X, n_neighbors, return_distance=True)

    def save(self, output_location=None):
        self.register_pkl_object(self.index, "index")
        self._save(output_location=output_location)

    def load(self, input_location=None):
        self._load(input_location=input_location)
        self.index = self.pkl_registry[0][0]
