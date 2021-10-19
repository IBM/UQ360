
import numpy as np
from uq360.utils.calibrators.calibrator import Calibrator

class ShiftCalibrator(Calibrator):
    '''
       Calibrator based on a fitted constant shift.
     '''
    def __init__(self):
        super(ShiftCalibrator, self).__init__()
        self.shift_value = 0

    @classmethod
    def name(cls):
        return ('shift')

    def fit(self, predicted_confidences, labels):
        predictions = np.where(predicted_confidences > 0.5, 1, 0)
        predicted_accuracy = np.mean(predictions)
        actual_accuracy = np.mean(labels)
        self.shift_value = actual_accuracy - predicted_accuracy
        self.fit_status = True

    def predict(self, predicted_confidences):
        predictions = np.where(predicted_confidences > 0.5, 1, 0)
        accuracy_predictions = predictions+self.shift_value
        return accuracy_predictions

    def save(self, output_location=None):
        self.register_json_object({"shift_value": self.shift_value}, 'shift_value')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        json_objs, _ = self.json_registry
        shift = json_objs[0]['shift_value']
        assert type(shift) == float
        self.shift_value = shift
        self.fit_status = True