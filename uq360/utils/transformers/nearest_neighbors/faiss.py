import faiss
from .base import BaseNearestNeighbors


class NearestNeighbors(BaseNearestNeighbors):
    def __int__(self):
        super(NearestNeighbors, self).__int__()
        self.index = None

    def fit(self, X, use_gpu=False, **kwargs):

        n_features = X.shape[1]
        self.index = faiss.IndexFlatL2(n_features, **kwargs)
        self.index.add(X)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        return self

    def transform(self, X, n_neighbors):
        distances, indices = self.index.search(X, n_neighbors)

        return distances, indices
