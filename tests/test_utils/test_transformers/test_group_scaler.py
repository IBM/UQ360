from unittest import TestCase

import numpy as np

from uq360.utils.transformers.group_scaler import GroupScaler


class TestGroupScaler(TestCase):
    def setUp(self) -> None:
        self.n_classes = 5
        self.n_per_class = 7
        self.d = 3
        self.scaler = GroupScaler()

        self.X = np.empty(shape=(self.n_classes * self.n_per_class, self.d))
        self.y = np.empty(shape=((self.n_classes * self.n_per_class,)))
        for i in range(self.n_classes):
            self.X[
                i * self.n_per_class : (i + 1) * self.n_per_class
            ] = np.random.normal(i, 0.2, size=(self.n_per_class, self.d))
            self.y[i * self.n_per_class : (i + 1) * self.n_per_class] = i

    def test_fit(self):
        try:
            self.scaler.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"GroupScaler.fit failed with {e}")

    def test_transform(self):
        self.scaler.fit(self.X, self.y)
        X_norm = self.scaler.transform(self.X, self.y)

        for i in range(self.n_classes):
            with self.subTest():
                idx = self.y == i
                self.assertTrue(np.allclose(np.mean(X_norm[idx], axis=0), 0.0))

    def test_fit_transform(self):
        X_norm = self.scaler.fit_transform(self.X, self.y)

        for i in range(self.n_classes):
            with self.subTest():
                idx = self.y == i
                self.assertTrue(np.allclose(np.mean(X_norm[idx], axis=0), 0.0))
