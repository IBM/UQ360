
import numpy as np
from uq360.utils.calibrators.calibrator import Calibrator


class ConfidenceBinsCalibrator(Calibrator):
    '''
    Calibrator based on histogram of model confidence scores.
    Recalibrates based on the sampling distribution from the (calibrator) training set. The (calibrator)
    train set accuracy for each set of samples defined by a confidence histogram bin is used
    as the recalibrated confidence value at inference time for any sample falling into that bin.
    '''
    def __init__(self):
        super(ConfidenceBinsCalibrator, self).__init__()
        self.predictor = {}

    @classmethod
    def name(cls):
        return ('confidence_bins')

    def get_confidence_dictionary(self, probs, ground_truth):
        conf_dict = {(a, a + 10): {'correct': 0, 'total': 0} for a in range(0, 100, 10)}

        for i in range(len(probs)):
            predicted_confidence = probs[i] * 100

            for k in conf_dict.keys():
                if predicted_confidence > k[0] and predicted_confidence <= k[1]:
                    conf_dict[k]['total'] += 1
                    if ground_truth[i]:
                        conf_dict[k]['correct'] += 1
                    break

        return conf_dict

    def fit(self, probs, ground_truth):
        conf_dict = self.get_confidence_dictionary(probs, ground_truth)

        conf_accuracies = {(a, a + 10): None for a in range(0, 100, 10)}
        conf_std = {(a, a + 10): None for a in range(0, 100, 10)}
        for k in conf_dict.keys():
            if conf_dict[k]['total'] != 0:
                conf_accuracies[k] = conf_dict[k]['correct'] / conf_dict[k]['total']
                conf_std[k] = (conf_accuracies[k] * (1 - conf_accuracies[k]) / conf_dict[k]['total']) ** 0.5
            else:
                conf_accuracies[k] = 0
                conf_std[k] = 0

        self.predictor = {}
        for k in conf_accuracies.keys():
            self.predictor[int((k[0]) / 10)] = {'mean': conf_accuracies[k], 'std': conf_std[k]}

        self.fit_status = True

    def predict(self, preds):
        accuracy_predictions = []
        for pred in preds:
            conf = pred * 100
            if conf == 100: conf = 99.99
            accuracy_predictions.append(self.predictor[int(conf / 10)]['mean'])

        return np.array(accuracy_predictions)

    def save(self, output_location=None):
        save_dictionary = {}
        for key, item in self.predictor.items():
            save_dictionary[str(key)] = item
        self.register_json_object(save_dictionary, 'confidence_dictionary')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)
        json_objs, _ = self.json_registry
        load_dictionary = json_objs[0]
        self.predictor = {}
        for key, item in load_dictionary.items():
            self.predictor[int(key)] = item
        self.fit_status = True
