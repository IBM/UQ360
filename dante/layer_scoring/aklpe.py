import numpy as np
from sklearn.model_selection import ShuffleSplit

from ..nearest_neighbors.base import BaseNearestNeighbors


class Aklpe:
    """Implementation of Averaged K nearest neighbors Localized P-value
    Estimation (aK_LPE) [1].

    [1] J. Qian and V. Saligrama, "New statistic in P-value estimation for
    anomaly detection," 2012 IEEE Statistical Signal Processing Workshop (SSP)
    """

    def __init__(
        self,
        nearest_neighbors: BaseNearestNeighbors,
        nearest_neighbors_kwargs={},
        n_neighbors: int = 50,
        n_bootstraps: int = 10,
        batch_size: int = 1,
        random_state: int = 123,
    ):
        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbors_kwargs = nearest_neighbors_kwargs
        self.n_bootstraps = n_bootstraps
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None):

        self.null_distribution = self._compute_null_distribution(X)

        self._fit_test_nn(X)

        return self

    def score(self, X: np.ndarray, y=None):

        # Compute g_scores
        test_g_stats = self._test_bootstrap(X)

        # Compute empirical p-values
        ranking = test_g_stats[:, np.newaxis] <= self.null_distribution
        p_values = ranking.sum(axis=1) / len(self.null_distribution)

        return test_g_stats, p_values

    def _g_statistic(self, X: np.ndarray, nearest_neighbors):

        lower_k = self.n_neighbors - (self.n_neighbors - 1) // 2
        upper_k = self.n_neighbors + self.n_neighbors // 2

        dist, idxs = nearest_neighbors.kneighbors(X, upper_k)
        g_stat = np.sort(dist, axis=1)[:, lower_k:]
        g_stat = np.mean(g_stat, axis=1)

        return g_stat

    def _compute_g_statistic(self, X, nearest_neighbors):

        scores = []
        for start_idx in range(0, len(X), self.batch_size):

            batch = X[start_idx : start_idx + self.batch_size]

            g_stat = self._g_statistic(batch, nearest_neighbors)

            scores.append(g_stat)

        scores = np.concatenate(scores)

        return scores

    def _fit_test_nn(self, X):

        nn_graphs = []
        for _ in range(self.n_bootstraps):

            rand_sub = np.random.randint(len(X), size=(len(X) // 2, ))
            nn_graph = self.nearest_neighbors().fit(
                X[rand_sub], **self.nearest_neighbors_kwargs
            )

            nn_graphs.append(nn_graph)

        self.nn_graphs = nn_graphs

    def _compute_null_distribution(self, X):

        split_generator = ShuffleSplit(
            n_splits=self.n_bootstraps, test_size=0.5, random_state=self.random_state
        )

        g_stats = []
        for s1, s2 in split_generator.split(X):

            # Fit nn graphs on len(X) // 2 instances
            s1_nn = self.nearest_neighbors().fit(X[s1], **self.nearest_neighbors_kwargs)
            s2_nn = self.nearest_neighbors().fit(X[s2], **self.nearest_neighbors_kwargs)

            # Compute g_stats on the other split
            s1_g_stats = self._compute_g_statistic(X[s1], s2_nn)
            s2_g_stats = self._compute_g_statistic(X[s2], s1_nn)

            del s1_nn
            del s2_nn

            split_g_stats = np.concatenate([s1_g_stats, s2_g_stats])
            idxs = np.concatenate([s1, s2])

            split_g_stats = split_g_stats[np.argsort(idxs)]

            g_stats.append(split_g_stats)

        g_stats = np.stack(g_stats).T

        return g_stats.mean(axis=1)

    def _test_bootstrap(self, X):

        g_stats = []
        for nn_graph in self.nn_graphs:

            g_stat = self._compute_g_statistic(X, nn_graph)

            g_stats.append(g_stat)

        g_stats = np.stack(g_stats).T

        return g_stats.mean(axis=1)
