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
    def __init__(self):
        super(FAISSNearestNeighbors, self).__init__()
        self.index = None
        self.X = None
        self.fit_kwargs = dict()

    @classmethod
    def name(cls):
        return 'faiss_nearest_neighbors'

    def fit(self, X, use_gpu=False, **kwargs):
        self.X = X.copy()
        self.fit_kwargs = dict(kwargs, use_gpu=use_gpu)

        n_features = X.shape[1]
        self.index = faiss.IndexFlatL2(n_features, **kwargs)
        self.index.add(X)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        return self

    def transform(self, X, n_neighbors):
        distances, indices = self.index.search(X, n_neighbors)

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

