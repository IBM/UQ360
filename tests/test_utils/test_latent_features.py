from unittest import TestCase

import torch

from tests.utils import PlusOne
from uq360.utils.latent_features import LatentFeatures


class TestLatentFeatures(TestCase):
    N = 10
    d = 2

    def setUp(self):
        self.first_plus_one = PlusOne()
        self.relu = torch.nn.ReLU()
        self.second_plus_one = PlusOne()
        self.m = torch.nn.Sequential(
            self.first_plus_one,
            self.relu,
            self.second_plus_one
        )

        self.X_ones = torch.ones((self.N, self.d), dtype=torch.float32)
        self.X_norm = torch.empty((self.N, self.d), dtype=torch.float32).normal_()

    def test_init(self):
        pass

    def test_get_first_plus(self):
        extractor = LatentFeatures(self.m, self.first_plus_one)
        with self.subTest():
            Z = extractor.extract(self.X_ones)[0]
            self.assertTrue(torch.all(Z == 2.))
        with self.subTest():
            Z = extractor.extract(self.X_norm)[0]
            self.assertTrue(torch.all(Z == (self.X_norm + 1.)))

    def test_get_relu(self):
        extractor = LatentFeatures(self.m, self.relu)
        with self.subTest():
            Z = extractor.extract(self.X_ones)[0]
            self.assertTrue(torch.all(Z == 2.))
        with self.subTest():
            Z = extractor.extract(self.X_norm)[0]
            expected_Z = self.X_norm + 1.
            expected_Z[expected_Z < 0] = 0.
            self.assertTrue(torch.all(Z == expected_Z))

    def test_multilayer(self):
        extractor = LatentFeatures(self.m, [self.first_plus_one, self.relu])
        with self.subTest():
            Z1, Z2 = extractor.extract(self.X_ones)
            self.assertTrue(torch.all(Z1 == 2.))
            self.assertTrue(torch.all(Z2 == 2.))
        with self.subTest():
            Z1, Z2 = extractor.extract(self.X_norm)
            self.assertTrue(torch.all(Z1 == (self.X_norm + 1.)))
            expected_Z2 = self.X_norm + 1.
            expected_Z2[expected_Z2 < 0] = 0.
            self.assertTrue(torch.all(Z2 == expected_Z2))


    def test_hook_cleanup(self):
        for layer in [self.first_plus_one, self.relu, self.second_plus_one]:
            with self.subTest():
                layer_hooks = layer._forward_hooks.copy()

                extractor = LatentFeatures(self.m, layer)
                extractor.extract(self.X_norm)

                self.assertDictEqual(layer_hooks, layer._forward_hooks)

    def test_post_processing_fn(self):
        def ppf(x):
            return x**2
        extractor = LatentFeatures(self.m, self.first_plus_one, post_processing_fn=ppf)
        for x in [self.X_ones, self.X_norm]:
            with self.subTest():
                z = extractor.extract(x)[0]
                self.assertTrue(
                    torch.all(z == (x+1)**2)
                )

