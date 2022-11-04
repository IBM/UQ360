try:
    import faiss
except ImportError as e:
    raise ImportError(
        f"This optional module depends on the uninstalled 'faiss' optional dependency: {__name__}"
        "Please note that faiss must be installed with conda."
    ) from e
import numpy as np

from .base import BaseNearestNeighbors


class FAISSNearestNeighbors(BaseNearestNeighbors):
    """Approximate nearest neighbor search using FAISS L2.

    Notes
    -----
    This requires the external dependency `faiss-cpu` or `faiss-gpu`.
    The CPU version can be installed through the unofficial PyPI version using
    `pip install faiss-cpu==1.6.5 ----no-cache`.

    The 'proper' procedure (which is nearly required for the GPU version) is to use `conda`,
    which works best by re-installing UQ360 and as many of its dependencies with `conda install`.
    """
    def __init__(self):
        super(FAISSNearestNeighbors, self).__init__()
        self.index = None
        self.X = None
        self.fit_kwargs = dict()

    @classmethod
    def name(cls):
        return 'faiss_nearest_neighbors'

    def fit(self, X, use_gpu=False, **kwargs):
        """Index a set of reference points

        Args:
            X: numpy array of reference points
            use_gpu: boolean, run on GPU?
            **kwargs: keyword arguments to be passed to faiss.IndexFlatL2

        Returns:
            self

        """
        self.X = X.copy().astype(np.float32)
        self.fit_kwargs = dict(kwargs, use_gpu=use_gpu)

        n_features = self.X.shape[1]
        self.index = faiss.IndexFlatL2(n_features, **kwargs)
        self.index.add(self.X)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

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
        distances, indices = self.index.search(X.astype(np.float32), n_neighbors)

        #FAISS reports the squared distance
        distances = np.sqrt(distances)

        return distances, indices

    def save(self, output_location=None):
        self.register_pkl_object(self.X, "X")
        self.register_pkl_object(self.fit_kwargs, "fit_kwargs")
        self._save(output_location=output_location)

    def load(self, input_location=None):
        self._load(input_location=input_location)

        objs, names = self.pkl_registry
        obj_map = dict(zip(names, objs))

        self.X = obj_map['X']
        self.fit_kwargs = obj_map['fit_kwargs']

