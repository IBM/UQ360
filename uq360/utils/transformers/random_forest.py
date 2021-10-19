
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from uq360.utils.transformers.feature_transformer import FeatureTransformer
from uq360.utils.transformers.confidence_top import ConfidenceTopTransformer
from uq360.utils.transformers.confidence_delta import ConfidenceDeltaTransformer
from uq360.utils.hpo_search import CustomRandomSearch


class RandomForestTransformer(FeatureTransformer):
    """Random forest shadow-model feature. This class trains a GBM model on the same train set as
    the input/baseline model. At inference time, the top class confidence and top - 2nd class
    confidence are used as the derived feature. """
    def __init__(self):
        super(RandomForestTransformer, self).__init__()
        self.random_seed = 42
        # specify parameters and distributions to sample from
        self.param_dist = {
                "n_estimators": sp_randint(5,100),
                "max_depth": [2,3,4,6, None],
                "min_samples_split": sp_randint(2, 6),
                "max_features": ["log2", "sqrt"]
            }
        self.model_params = {
                "random_state": self.random_seed,
                "verbose": 0
            }
        self.randomized_params = {
                "n_iter": 15,
                "n_jobs": -1,
                "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                "verbose": 0,
                "random_state": self.random_seed,
                "return_train_score": True,
                "progress_bar": False}

    @classmethod
    def name(cls):
        return ('random_forest')

    def fit(self, x, y):
        if x.shape[0] > 40000:
            x_train, _, y_train, _ = train_test_split(x, y, train_size=40000, random_state=self.random_seed)
            print()
            print("DOWNSAMPLING RANDOM FOREST FROM {} TO {}".format(x.shape[0], x_train.shape[0]))
        else:
            x_train = x
            y_train = y
        y_train = y_train.ravel()
        self.model = RandomForestClassifier()
        clf = CustomRandomSearch(self.model, self.param_dist, **self.randomized_params)
        clf.fit(x_train, y_train)
        self.model = clf.best_estimator_
        self.register_pkl_object(self.model, 'model')
        self.fit_status = True

    def transform(self, x, predictions):
        assert self.fit_status
        preds = self.model.predict_proba(x)
        top = ConfidenceTopTransformer().transform(x, preds).reshape(-1,1)
        delta = ConfidenceDeltaTransformer().transform(x, preds).reshape(-1,1)
        result = np.concatenate([top, delta], axis=1)
        return result

    def save(self, output_location=None):
        self.register_pkl_object(self.model, 'model')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        self.model = self.pkl_registry[0][0]
        assert type(self.model) == RandomForestClassifier
        self.fit_status = True
