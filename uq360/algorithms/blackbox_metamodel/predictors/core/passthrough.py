
import numpy as np
from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor

"""
Dummy performance predictor which simply passes the predicted class confidence of the base/input model as its 
own prediction. This performance predictor does not have a method to quantify its own uncertainty, so the uncertainty 
values are zero.  
"""
class PassthroughPredictor(PerfPredictor):
    def __init__(self, calibrator=None):
        self.predictor = {}
        calibrator = None
        super(PassthroughPredictor, self).__init__(calibrator)

    @classmethod
    def name(cls):
        return ('passthrough')

    def fit(self, x_test_unprocessed, x_test, y_test):
        self.fit_status = True

    def predict(self, X_unprocessed, X):
        assert self.fit_status
        assert 'confidence_top' in X.keys()
        preds = X['confidence_top'].values

        output = {'confidences': np.array(preds), 'uncertainties': np.zeros(preds.shape)}
        return output

    def save(self, output_location):
        pass

    def load(self, input_location):
        pass
