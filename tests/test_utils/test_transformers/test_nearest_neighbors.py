from unittest import TestCase

import numpy as np

from uq360.utils.transformers.nearest_neighbors.exact import ExactNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.faiss import FAISSNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.pynndescent import PyNNDNearestNeighbors


class KNNTester:
    n_classes = 3
    n_per_class = 50
    d = 3

    def get_perfect_X(self):
        Xs = []
        for i in range(self.n_classes):
            Xs.append(i * np.ones(shape=(self.n_per_class, self.d), dtype=np.float32))

        return np.concatenate(Xs, axis=0)

    def get_X_query(self):
        Xs = []
        for i in range(self.n_classes):
            Xs.append(i * np.ones(shape=(1, self.d), dtype=np.float32))

        return np.concatenate(Xs, axis=0)

    def test_instantiate(self):
        try:
            self.knn_class()
        except Exception as e:
            self.fail(f"{self.__class__.__name__} raised exception {e} when instantiating")

    def test_fit(self):
        knn = self.knn_class()

        X = self.get_perfect_X()

        try:
            knn.fit(X, **self.fit_kwargs)
        except Exception as e:
            self.fail(f"{self.__class__.__name__}.fit raised exception {e}")

    def test_predict(self):
        knn = self.knn_class()
        X = self.get_perfect_X()

        knn.fit(X, **self.fit_kwargs)

        try:
            X_query = self.get_X_query()
            dist, idx = knn.transform(X_query, **self.transform_kwargs)
        except Exception as e:
            self.fail(f"{self.__class__.__name__}.fit raised exception {e}")

        self.assertTrue(np.allclose(dist,0))

        for i in range(self.n_classes):
            self.assertSetEqual(set(idx[i]), {i * self.n_per_class + k for k in range(self.n_per_class)})


class TestExactKNN(TestCase, KNNTester):
    knn_class = ExactNearestNeighbors
    fit_kwargs = {'n_neighbors': KNNTester.n_per_class}
    transform_kwargs = {'n_neighbors': KNNTester.n_per_class}


class TestPynnDescentKNN(TestCase, KNNTester):
    knn_class = PyNNDNearestNeighbors
    fit_kwargs = {'n_neighbors': KNNTester.n_per_class}
    transform_kwargs = {'n_neighbors': KNNTester.n_per_class}


class TestFaissKNN(TestCase, KNNTester):
    knn_class = FAISSNearestNeighbors
    fit_kwargs = {}
    transform_kwargs = {'n_neighbors': KNNTester.n_per_class}
