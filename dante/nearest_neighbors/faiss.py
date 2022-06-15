import faiss


class NearestNeighbors:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, use_gpu=False):

        n_features = X.shape[1]
        self.index = faiss.IndexFlatL2(n_features)
        self.index.add(X)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        return self

    def kneighbors(self, X, n_neighbors):
        distances, indices = self.index.search(X, n_neighbors)

        return distances, indices
