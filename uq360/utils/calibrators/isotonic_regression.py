
import numpy as np
from sklearn.isotonic import IsotonicRegression
from uq360.utils.calibrators.calibrator import Calibrator

class IsotonicRegressionCalibrator(Calibrator):
    '''
    Calibrator based on isotonic regression procedure.
    This calibrator finds the best piecewise-constant, monotonic function of the confidences
    to recalibrate to represent the probability of a correct classification.
    '''
    def __init__(self):
        super(IsotonicRegressionCalibrator, self).__init__()
        self.isotonic_regressor = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')

    @classmethod
    def name(cls):
        return ('isotonic_regression')

    def fit(self, predicted_confidences, labels):
        n_plus = np.sum(labels == 1)
        n_minus = np.sum(labels == 0)
        isotonic_target = np.where(labels == 0, 1.0 / (n_minus + 2.0), (n_plus + 1.0) / (n_plus + 2.0))
        try:
            self.isotonic_regressor.fit(predicted_confidences, isotonic_target)
        except:
            self.isotonic_regressor.fit(predicted_confidences.astype(float), isotonic_target.astype(float))
        self.fit_status = True

    def predict(self, predicted_confidences):
        accuracy_predictions = self.isotonic_regressor.predict(predicted_confidences)
        return accuracy_predictions

    def save(self, output_location=None):
        self.register_pkl_object(self.isotonic_regressor, 'iso')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        pkl_objs, _ = self.pkl_registry
        iso = pkl_objs[0]
        assert type(iso) == IsotonicRegression
        self.isotonic_regressor = iso
        self.fit_status = True
