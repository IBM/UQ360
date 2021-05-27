from collections import namedtuple

from uq360.algorithms.posthocuq import PostHocUQ
from uq360.utils.misc import form_D_for_auucc
from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve


class UCCRecalibration(PostHocUQ):
    """ Recalibration a regression model to specified operating point using Uncertainty Characteristics Curve.
    """

    def __init__(self, base_model):
        """
        Args:
            base_model: pretrained model to be recalibrated.
        """
        super(UCCRecalibration).__init__()
        self.base_model = self._process_pretrained_model(base_model)
        self.ucc = None

    def get_params(self, deep=True):
        return {"base_model": self.base_model}

    def _process_pretrained_model(self, base_model):
        return base_model

    def fit(self, X, y):
        """
        Fit the Uncertainty Characteristics Curve.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values

        Returns:
            self

        """
        y_pred_mean, y_pred_lower, y_pred_upper = self.base_model.predict(X)[:3]
        bwu = y_pred_upper - y_pred_mean
        bwl = y_pred_mean - y_pred_lower
        self.ucc = UncertaintyCharacteristicsCurve()
        self.ucc.fit(form_D_for_auucc(y_pred_mean, bwl, bwu), y.squeeze())

        return self

    def predict(self, X, missrate=0.05):
        """
        Generate prediction and uncertainty bounds for data X.

        Args:
            X: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
            missrate: desired missrate of the new operating point, set to 0.05 by default.

        Returns:
            namedtuple: A namedtupe that holds

            y_mean: ndarray of shape (n_samples, [n_output_dims])
                Mean of predictive distribution of the test points.
            y_lower: ndarray of shape (n_samples, [n_output_dims])
                Lower quantile of predictive distribution of the test points.
            y_upper: ndarray of shape (n_samples, [n_output_dims])
                Upper quantile of predictive distribution of the test points.
        """
        C = self.ucc.get_specific_operating_point(req_y_axis_value=missrate, vary_bias=False)
        new_scale = C['modvalue']

        y_pred_mean, y_pred_lower, y_pred_upper = self.base_model.predict(X)[:3]
        bwu = y_pred_upper - y_pred_mean
        bwl = y_pred_mean - y_pred_lower

        if C['operation'] == 'bias':
            calib_y_pred_upper = y_pred_mean + (new_scale + bwu)  # lower bound width
            calib_y_pred_lower = y_pred_mean - (new_scale + bwl)  # Upper bound width
        else:
            calib_y_pred_upper = y_pred_mean + (new_scale * bwu)  # lower bound width
            calib_y_pred_lower = y_pred_mean - (new_scale * bwl)  # Upper bound width

        Result = namedtuple('res', ['y_mean', 'y_lower', 'y_upper'])
        res = Result(y_pred_mean, calib_y_pred_lower, calib_y_pred_upper)

        return res
