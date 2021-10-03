import numpy as np
from uq360.utils.calibrators.calibrator import Calibrator

class LinearExtrapolationCalibrator(Calibrator):
    '''
    Calibrator based on a fitted linear transformation of the confidence scores.
    '''
    def __init__(self):
        super(LinearExtrapolationCalibrator, self).__init__()
        self.slope = 0
        self.intercept = 0

    @classmethod
    def name(cls):
        return ('linear_extrapolation')

    def lin_equ(self, l1, l2):
        if l2[0] == l1[0]:
            return 0, l1[0]
        m = (l2[1] - l1[1]) / (l2[0] - l1[0])
        c = (l2[1] - (m * l2[0]))
        return m, c

    def fit(self, predicted_confidences, labels):
        predictions = np.where(predicted_confidences > 0.5, 1, 0)
        mean_correct = np.mean(predictions[labels==1])
        mean_incorrect = np.mean(predictions[labels==0])
        self.slope, self.intercept = self.lin_equ([mean_correct, 1], [mean_incorrect, 0])
        self.fit_status = True

    def predict(self, predicted_confidences):
        assert self.fit_status is True
        predictions = np.where(predicted_confidences > 0.5, 1, 0)
        accuracy_predictions = (predictions*self.slope) + self.intercept
        return accuracy_predictions
