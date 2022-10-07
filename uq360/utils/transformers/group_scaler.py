import numpy as np

from .feature_transformer import FeatureTransformer


class GroupScaler(FeatureTransformer):
    @classmethod
    def name(cls):
        return 'group_scaler'

    def __init__(self, class_means=None):
        super(GroupScaler, self).__init__()
        self.class_means = class_means

    def fit(self, X, y):
        self.class_means = {
            loop_y: np.mean(X[loop_y == y], axis=0) for loop_y in set(list(y))
        }

        return self

    def transform(self, X, y):
        old_idxs = []
        X_norm = []
        for loop_y in set(list(y)):
            mask_idxs = (y == loop_y).nonzero()

            temp_X = X[mask_idxs] - self.class_means[loop_y]

            X_norm.extend(temp_X)
            old_idxs.extend(mask_idxs)

        old_idxs = np.concatenate(old_idxs)
        X_norm = np.stack(X_norm)[old_idxs.argsort()]

        return X_norm

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def save(self, output_location=None):
        self.register_pkl_object("class_means", self.class_means)
        self._save(output_location=output_location)

    def load(self, input_location=None):
        self._load(input_location=input_location)
        self.class_means = self.pkl_registry[0][0]