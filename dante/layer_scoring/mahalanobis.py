import torch

import numpy as np

from ..preprocessing.group_scaler import GroupScaler
from sklearn.covariance import EmpiricalCovariance


class Mahalanobis:
    """Implementation of Mahalanobis Adversarial/Out-of-distribution detector [1].

    [1] "A Simple Unified Framework for Detecting Out-of-Distribution Samples and
    Adversarial Attacks", K. Lee et al., NIPS 2018.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.scaler = GroupScaler()

        X = self.scaler.fit_transform(X, y)

        self._set_covariance(X)

        return self

    def score(self, X, y=None) -> np.ndarray:

        all_scores = []
        class_means = self.scaler.class_means.values()
        for mean in class_means:
            X_centered = X - mean
            scores = np.matmul(X_centered, self.precision)
            scores = np.matmul(scores, X_centered.T)
            scores = scores.diag()
            all_scores.append(-scores)

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
