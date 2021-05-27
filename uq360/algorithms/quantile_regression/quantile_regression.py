from collections import namedtuple

from sklearn.ensemble import GradientBoostingRegressor

from uq360.algorithms.builtinuq import BuiltinUQ


class QuantileRegression(BuiltinUQ):
    """Quantile Regression uses quantile loss and learns two separate models for the upper and lower quantile
    to obtain the prediction intervals.
    """

    def __init__(self, model_type="gbr", config=None):
        """
        Args:
            model_type: The base model used for predicting a quantile. Currently supported values are [gbr].
                gbr is sklearn GradientBoostingRegressor.
            config: dictionary containing the config parameters for the model.
        """

        super(QuantileRegression).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = {}
        if "alpha" not in self.config:
            self.config["alpha"] = 0.95
        if model_type == "gbr":
            self.model_type = model_type
            self.model_mean = GradientBoostingRegressor(
                loss='ls',
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                learning_rate=self.config["learning_rate"],
                min_samples_leaf=self.config["min_samples_leaf"],
                min_samples_split=self.config["min_samples_split"]
            )
            self.model_upper = GradientBoostingRegressor(
                loss='quantile',
                alpha=self.config["alpha"],
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                learning_rate=self.config["learning_rate"],
                min_samples_leaf=self.config["min_samples_leaf"],
                min_samples_split=self.config["min_samples_split"]
            )
            self.model_lower = GradientBoostingRegressor(
                loss='quantile',
                alpha=1.0 - self.config["alpha"],
                n_estimators=self.config["n_estimators"],
                max_depth=self.config["max_depth"],
                learning_rate=self.config["learning_rate"],
                min_samples_leaf=self.config["min_samples_leaf"],
                min_samples_split=self.config["min_samples_split"])

        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        return {"model_type": self.model_type, "config": self.config}

    def fit(self, X, y):
        """ Fit the Quantile Regression model.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """

        self.model_mean.fit(X, y)
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)

        return self

    def predict(self, X):
        """
        Obtain predictions for the test points.

        In addition to the mean and lower/upper bounds, also returns epistemic uncertainty (return_epistemic=True)
        and full predictive distribution (return_dists=True).

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
        """

        y_mean = self.model_mean.predict(X)
        y_lower = self.model_lower.predict(X)
        y_upper = self.model_upper.predict(X)

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_mean, y_lower, y_upper)

        return res
