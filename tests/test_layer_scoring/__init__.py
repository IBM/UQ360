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
            X = np.random.normal(size=(100, 2))
            scorer = self.ScorerClass(**self.scorer_kwargs)
            scorer.fit(X)
        except Exception as e:
            self.fail(f"{self.ScorerClass.__name__}.fit failed with {e}")

    def test_fit_latent(self):
        try:
            X = np.random.normal(size=(100, 5))
            scorer = self.ScorerClass(
                model=self.model, layer=self.layer, **self.scorer_kwargs
            )
            scorer.fit(X)
        except Exception as e:
            self.fail(f"{self.ScorerClass.__name__}.fit failed with {e}")
