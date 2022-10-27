from tests.test_layer_scoring import LatentScorerTester
from uq360.algorithms.layer_scoring.mahalanobis import MahalanobisScorer


class TestMahalanobisScorer(LatentScorerTester):
    ScorerClass = MahalanobisScorer
    fit_with_y = True
    scorer_kwargs = {}
