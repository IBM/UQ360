
import numpy as np
from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor


"""
Performance predictor based on confidence binning. The highest predicted class confidence from the base/input model is 
recorded for each sample. These confidences are then binned in a histogram, and the average and standard deviation of 
each bin are computed. 

In the prediction step, the predictor first finds the bin in this histogram corresponding to the top confidence 
of the base model prediction. The performance predictor confidence for this sample is then the average value recorded 
for this bin, and the uncertainty for this sample is the associated standard deviation. 
"""
class ConfidencePredictor(PerfPredictor):
    def __init__(self, calibrator=None):
        self.predictor = {}
        calibrator = None
        super(ConfidencePredictor, self).__init__(calibrator)

    @classmethod
    def name(cls):
        return ('confidence')

    def get_conf_dict(self, top_confidences, y_test):
        # confidence bins
        conf_dict = {(a, a + 10): {'correct': 0, 'total': 0} for a in range(0, 100, 10)}
        for i in range(len(top_confidences)):
            pred_conf = top_confidences[i] * 100

            for k in conf_dict.keys():
                if pred_conf > k[0] and pred_conf < k[1]:
                    # total num of examples that have confidence within the range (say 10-20)
                    conf_dict[k]['total'] += 1

                    if y_test[i]:
                        #  number of examples that have confidence within this range
                        #  and have correct predictions
                        conf_dict[k]['correct'] += 1

                    break
        return conf_dict

    def fit(self, x_test_unprocessed, x_test, y_test):
        try:
            assert 'confidence_top' in x_test.keys()
        except:
            raise Exception("confidence predictor must be called with 'confidence_top' transformer. ")
        
        probs = x_test['confidence_top'].values

        conf_dict = self.get_conf_dict(probs, y_test)

        conf_accs = {(a, a + 10): None for a in range(0, 100, 10)}
        conf_std = {(a, a + 10): None for a in range(0, 100, 10)}

        # for every confidence bin, find the accuracy
        # for example, if total is 10 and 5 is correct then acc is 50% in that bin

        for k in conf_dict.keys():
            if conf_dict[k]['total'] != 0:
                conf_accs[k] = conf_dict[k]['correct'] / conf_dict[k]['total']
                conf_std[k] = (conf_accs[k] * (1 - conf_accs[k]) / conf_dict[k]['total']) ** 0.5
            else:
                conf_accs[k] = 0
                conf_std[k] = 0

        self.predictor = {}
        for k in conf_accs.keys():
            # mean -> acc
            self.predictor[int((k[0]) / 10)] = {'mean': conf_accs[k], 'std': conf_std[k]}

        #  {(1, 11): {'correct': 0, 'total': 0}, (11, 21): {'correct': 0, 'total': 0},
        #  (21, 31): {'correct': 0, 'total': 0}, (31, 41): {'correct': 0, 'total': 0},
        #  (41, 51): {'correct': 0, 'total': 0}, (51, 61): {'correct': 0, 'total': 0},
        #  (61, 71): {'correct': 0, 'total': 0}, (71, 81): {'correct': 0, 'total': 0},
        #  (81, 91): {'correct': 1685, 'total': 1888}, (91, 101): {'correct': 0, 'total': 0}}


        #  self.predictor looks like the below
        #  {0: {'mean': 0, 'std': 0}, 1: {'mean': 0, 'std': 0}, 2: {'mean': 0, 'std': 0}, 3: {'mean': 0, 'std': 0},
        #  4: {'mean': 0, 'std': 0}, 5: {'mean': 0, 'std': 0}, 6: {'mean': 0, 'std': 0}, 7: {'mean': 0, 'std': 0},
        #  8: {'mean': 0.892478813559322, 'std': 0.0071292687519874795}, 9: {'mean': 0, 'std': 0}}

        # if the original model is 89% accurate on test data between 80%-90% confidence
        # in other words, when prod data is between 80-90% confidence, u will say that the original
        # model will be 89% accurate
        self.fit_status = True

    def predict(self, X_unprocessed, X):
        # TODO: add some sanity checks for X
        assert self.fit_status
        assert 'confidence_top' in X.keys()
        preds = X['confidence_top'].values

        accuracy_predictions = []
        standard_deviation = []
        for pred in preds:
            conf = pred * 100
            if conf == 100: conf = 99.99

            # if np.argmax(pred)
            # look up from self.predictor
            accuracy_predictions.append(self.predictor[int(conf / 10)]['mean'])
            standard_deviation.append(self.predictor[int(conf / 10)]['std'])
        output = {'confidences': np.array(accuracy_predictions), 'uncertainties': np.array(standard_deviation)}
        return output

    def save(self, output_location):
        output_json = {}
        for key, item in self.predictor.items():
            output_json[str(key)] = item
        self.register_json_object(output_json, 'output_dict')
        self._save(output_location)
    
    def load(self, input_location):
        self._load(input_location)
        json_objs, json_names = self.json_registry
        for obj, name in zip(json_objs, json_names):
            if name == 'output_dict':
                self.predictor = {}
                for key, item in obj.items():
                    self.predictor[int(key)] = item
        self.fit_status = True
