from ..feature_transformer import FeatureTransformer


class BaseNearestNeighbors(FeatureTransformer):
    """Parent class for feature transformers performing nearest neighbor searches.
    The predict method returns a tuple with a tensor of distances to the kNN and
    and tensor of indices of these kNN.
    """
    pass
