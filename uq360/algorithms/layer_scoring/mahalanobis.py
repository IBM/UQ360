import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance

from uq360.algorithms.layer_scoring.latent_scorer import LatentScorer
from uq360.utils.transformers.group_scaler import GroupScaler


class MahalanobisScorer(LatentScorer):
    """Implementation of the Mahalanobis Adversarial/Out-of-distribution detector [1].

    [1] "A Simple Unified Framework for Detecting Out-of-Distribution Samples and
    Adversarial Attacks", K. Lee et al., NIPS 2018.
    """

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
        super(MahalanobisScorer, self).__init__(model=model, layer=layer)
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Register data X and class labels y as in-distribution data"""
        return super(MahalanobisScorer, self).fit(X, y)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = GroupScaler()
        X = self.scaler.fit_transform(X, y)
        self._set_covariance(X)

        return self

    def predict(self, X: np.ndarray):
        """Compute the Mahalanobis distance between query data X and the in-distribution data classes"""
        return super(MahalanobisScorer, self).predict(X)

    def _predict(self, X) -> np.ndarray:
        all_scores = []
        class_means = self.scaler.class_means.values()
        for mean in class_means:
            X_centered = X - mean
            scores = np.matmul(X_centered, self.precision)
            scores = np.matmul(scores, X_centered.T)
            scores = np.diag(scores)
            all_scores.append(scores)

        all_scores = np.stack(all_scores).T

        return np.max(all_scores, axis=1)

    def _set_covariance(self, X, lib="sklearn", bias=True, rowobs=True):

        if lib == "sklearn":

            cov = EmpiricalCovariance().fit(X)

            self.covariance = cov.covariance_
            self.precision = cov.precision_

        elif lib == "torch":

            if rowobs:
                X = X.t()

            N = X.shape[0]

            cov = X.cov()

            if bias:
                cov = cov * (N - 1) / N

            self.covariance = cov
            self.precision = torch.linalg.pinv(cov, hermitian=True)

        else:
            raise NotImplementedError()

    def _process_pretrained_model(self, X, model, layer):
        pass

    def get_params(self):
        """This method is parameterless"""
        pass
