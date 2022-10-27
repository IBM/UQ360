import unittest
from typing import Type
from unittest import TestCase

import numpy as np
from torch import nn

from tests.utils import PlusOne
from uq360.algorithms.layer_scoring.latent_scorer import LatentScorer


class LatentScorerTester(TestCase):
    ScorerClass: Type[LatentScorer]
    scorer_kwargs: dict
    predict_kwargs: dict
    fit_with_y: bool

    @classmethod
    def setUpClass(cls):
        if cls is LatentScorerTester:
            raise unittest.SkipTest("Skipping base class LatentScorerTester")
        super(LatentScorerTester, cls).setUpClass()

    def setUp(self) -> None:
        self.p1 = PlusOne()
        self.relu = nn.ReLU()
        self.model = nn.Sequential(self.p1, self.relu)
        self.layer = self.p1

    @staticmethod
    def verify_latent_scorer_get_latents_case(
        ScorerClass: Type[LatentScorer],
        *,
        model: nn.Module = None,
        layer: nn.Module = None,
        X: np.array,
        expected_z: np.array,
        **scorer_kwargs,
    ):
        scorer = ScorerClass(model=model, layer=layer, **scorer_kwargs)

        z = scorer.get_latents(X)

        return np.allclose(z, expected_z)

    def test_latent_scorer_get_latents(self):
        with self.subTest("No model extraction"):
            X = np.random.normal(size=(10, 5))
            self.assertTrue(
                self.verify_latent_scorer_get_latents_case(
                    self.ScorerClass, X=X, expected_z=X, **self.scorer_kwargs
                )
            )

        with self.subTest("Model layer extraction"):
            X = np.random.normal(size=(10, 5))

            expected_z = X + 1.0
            self.assertTrue(
                self.verify_latent_scorer_get_latents_case(
                    self.ScorerClass,
                    model=self.model,
                    layer=self.layer,
                    X=X,
                    expected_z=expected_z,
                    **self.scorer_kwargs,
                )
            )

    def test_fit_direct(self):
        try:
            fit_data = dict(X=np.random.normal(size=(100, 2)))
            if self.fit_with_y:
                fit_data.update(y=np.random.randint(0, 3, size=100))

            scorer = self.ScorerClass(**self.scorer_kwargs)
            scorer.fit(**fit_data)
        except Exception as e:
            self.fail(f"{self.ScorerClass.__name__}.fit failed with {e}")

    def test_fit_latent(self):
        try:
            fit_data = dict(X=np.random.normal(size=(100, 2)))
            if self.fit_with_y:
                fit_data.update(y=np.random.randint(0, 3, size=100))

            scorer = self.ScorerClass(
                model=self.model, layer=self.layer, **self.scorer_kwargs
            )
            scorer.fit(**fit_data)
        except Exception as e:
            self.fail(f"{self.ScorerClass.__name__}.fit failed with {e}")

    def test_predict_direct(self):
        n_per_class = 50
        d = 3
        X1 = np.random.normal(1.0, 0.1, size=(n_per_class, d)).astype(np.float32)
        X0 = np.random.normal(0.0, 0.1, size=(n_per_class, d)).astype(np.float32)
        fit_data = dict(X=np.concatenate([X0, X1], axis=0))
        if self.fit_with_y:
            y1 = np.ones(shape=(n_per_class,)).astype(np.float32)
            y0 = np.zeros(shape=(n_per_class,)).astype(np.float32)
            fit_data.update(y=np.concatenate([y0, y1]))

        scorer = self.ScorerClass(**self.scorer_kwargs)
        scorer.fit(**fit_data)

        n_query = 3
        X_query_0 = 0.0 * np.ones((n_query, d)).astype(np.float32)
        X_query_20 = 20.0 * np.ones((n_query, d)).astype(np.float32)

        d0 = np.mean(scorer.predict(X_query_0))
        d20 = np.mean(scorer.predict(X_query_20))

        self.assertGreater(d20, d0)
