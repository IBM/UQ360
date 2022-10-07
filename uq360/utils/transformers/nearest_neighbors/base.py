from abc import ABC, abstractmethod


class BaseNearestNeighbors(ABC):
    @abstractmethod
    def fit(self, X, **kwargs):
        pass

    @abstractmethod
    def transform(self, X, n_neighbors):
        pass
