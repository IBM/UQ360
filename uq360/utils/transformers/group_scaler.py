import numpy as np


class GroupScaler:
    def __init__(self):
        super(GroupScaler, self).__init__()
        self.class_means = None

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
