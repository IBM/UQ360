
import numpy as np
from uq360.calibrators.calibrator import Calibrator


class ConfidenceBinsCalibrator(Calibrator):
    def __init__(self):
        super(ConfidenceBinsCalibrator, self).__init__()
        self.predictor = {}

    @classmethod
    def name(cls):
        return ('confidence_bins')

    def get_confidence_dictionary(self, probs_metamodel, metamodel_ground_truth):
        conf_dict = {(a, a + 10): {'correct': 0, 'total': 0} for a in range(0, 100, 10)}

        for i in range(len(probs_metamodel)):
            pred_conf = probs_metamodel[i] * 100

            for k in conf_dict.keys():
                if pred_conf > k[0] and pred_conf <= k[1]:
                    conf_dict[k]['total'] += 1
                    if metamodel_ground_truth[i]:
                        conf_dict[k]['correct'] += 1
                    break

        return conf_dict

    def fit(self, probs_metamodel, metamodel_ground_truth):
        conf_dict = self.get_confidence_dictionary(probs_metamodel, metamodel_ground_truth)

        conf_accs = {(a, a + 10): None for a in range(0, 100, 10)}
        conf_std = {(a, a + 10): None for a in range(0, 100, 10)}
        for k in conf_dict.keys():
            if conf_dict[k]['total'] != 0:
                conf_accs[k] = conf_dict[k]['correct'] / conf_dict[k]['total']
                conf_std[k] = (conf_accs[k] * (1 - conf_accs[k]) / conf_dict[k]['total']) ** 0.5
            else:
                conf_accs[k] = 0
                conf_std[k] = 0

        self.predictor = {}
        for k in conf_accs.keys():
            self.predictor[int((k[0]) / 10)] = {'mean': conf_accs[k], 'std': conf_std[k]}

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
