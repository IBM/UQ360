import abc
from abc import ABC

import torch

from uq360.algorithms.posthocuq import PostHocUQ
from uq360.utils.latent_features import LatentFeatures


class LatentScorer(PostHocUQ, ABC):
    def __init__(self, model=None, layer=None):
        super(LatentScorer, self).__init__()
        if model is not None and layer is not None:
            assert isinstance(layer, torch.nn.Module)
            self.extractor = LatentFeatures(model=model, layer=layer)
        else:
            self.extractor = None

    def get_latents(self, X):
        if self.extractor is not None:
            X = self.extractor.extract(torch.tensor(X))[0].numpy()
        return X

    @abc.abstractmethod
    def _fit(self, X, *args, **kwargs):
        pass

    def fit(self, X, *args, **kwargs):
        X = self.get_latents(X)
        return self._fit(X, *args, kwargs)

    @abc.abstractmethod
    def _predict(self, X, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        X = self.get_latents(X)
        return self._predict(X, *args, **kwargs)
