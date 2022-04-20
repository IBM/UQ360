import pynndescent

class NearestNeighbors():

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def fit(self, X, ):
        self.index = pynndescent.NNDescent(X, **self.kwargs)
        self.index.prepare()

        return self

    def kneighbors(self, X, n_neighbors):
        indices, distances = self.index.query(X, n_neighbors)

        return distances, indices