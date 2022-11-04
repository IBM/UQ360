import abc
from abc import ABC

import torch

from uq360.algorithms.posthocuq import PostHocUQ
from uq360.utils.latent_features import LatentFeatures


class LatentScorer(PostHocUQ, ABC):
    """PostHoc Uncertainty Quantification base class for analyzing latent representations of data from a model"""
    def __init__(self, model=None, layer=None):
        """

        Args:
            model: torch Module to analyze
            layer: layer (torch Module) inside the model whose output is to be analyzed
        Notes:
            The model and layer arguments are optional.
            If no model or layer is provided, it is expected that the inputs are already latent vectors.
            If both a model and layers are provided, inputs are expected to be model inputs
            to be mapped to latent vectors.
        """
        super(LatentScorer, self).__init__()
        if model is not None and layer is not None:
            assert isinstance(layer, torch.nn.Module)
            self.extractor = LatentFeatures(model=model, layer=layer)
        else:
            self.extractor = None

    def get_latents(self, X):
        if self.extractor is not None:
            X = self.extractor.extract(torch.tensor(X))[0]
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return X

    @abc.abstractmethod
    def _fit(self, X, *args, **kwargs):
        pass

    def fit(self, X, *args, **kwargs):
        X = self.get_latents(X)
        return self._fit(X, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, X, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        X = self.get_latents(X)
        return self._predict(X, *args, **kwargs)
