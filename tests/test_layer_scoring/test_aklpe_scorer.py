from tests.test_layer_scoring import LatentScorerTester
from uq360.algorithms.layer_scoring.aklpe import AKLPEScorer
from uq360.utils.transformers.nearest_neighbors.exact import ExactNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.faiss import FAISSNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.pynndescent import PyNNDNearestNeighbors


class TestAKLPEScorerExact(LatentScorerTester):
    ScorerClass = AKLPEScorer
    fit_with_y = False
    scorer_kwargs = {"nearest_neighbors": ExactNearestNeighbors, "n_neighbors": 5}


class TestAKLPEScorerPyNNDescent(LatentScorerTester):
    ScorerClass = AKLPEScorer
    fit_with_y = False
    scorer_kwargs = {"nearest_neighbors": PyNNDNearestNeighbors, "n_neighbors": 5}


class TestAKLPEScorerFAISS(LatentScorerTester):
    ScorerClass = AKLPEScorer
    fit_with_y = False
    scorer_kwargs = {"nearest_neighbors": FAISSNearestNeighbors, "n_neighbors": 5}
