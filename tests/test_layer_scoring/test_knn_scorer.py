from tests.test_layer_scoring import LatentScorerTester
from tests.utils import base_test_case
from uq360.algorithms.layer_scoring.knn import KNNScorer
from uq360.utils.transformers.nearest_neighbors.exact import ExactNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.faiss import FAISSNearestNeighbors
from uq360.utils.transformers.nearest_neighbors.pynndescent import PyNNDNearestNeighbors



class TestKNNScorerExact(LatentScorerTester):
    ScorerClass = KNNScorer
    scorer_kwargs = {"nearest_neighbors": ExactNearestNeighbors}


class TestKNNScorerPyNNDescent(LatentScorerTester):
    ScorerClass = KNNScorer
    scorer_kwargs = {"nearest_neighbors": PyNNDNearestNeighbors}


class TestKNNScorerFAISS(LatentScorerTester):
    ScorerClass = KNNScorer
    scorer_kwargs = {"nearest_neighbors": FAISSNearestNeighbors}
