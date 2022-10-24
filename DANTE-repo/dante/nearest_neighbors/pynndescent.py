import pynndescent


class NearestNeighbors:
    def fit(self, X, **kwargs):
        self.index = pynndescent.NNDescent(X, kwargs)
        self.index.prepare()

        return self

    def kneighbors(self, X, n_neighbors):
        indices, distances = self.index.query(X, n_neighbors)

        return distances, indices
