
import numpy as np

from uq360.base import Base
from uq360.utils.batch_features.histogram_utilities import compute_average_entropy
from uq360.utils.significance_test import SignificanceTester


class BlackboxFeature(Base):
    """Batchwise-feature based only on the output of the performance-predictor (meta-model to predict accuracy of the
    base/input model). """
    def __init__(self, train_labels=None):
        assert train_labels is not None
        self.label_dict = {}
        for i, l in enumerate(train_labels):
            self.label_dict[i] = l

    def convert_to_labels(self, vec):
        new_vec = [self.label_dict[x] for x in vec]
        return np.array(new_vec)

    def extract_payload(self, base_predictions: np.ndarray, y_test: np.ndarray):
        return {}

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        raise NotImplementedError("'extract_blackbox_feature' must be implemented for all subclasses")

    @classmethod
    def instance(cls, subtype_name=None, **params):
        subtype_name = subtype_name

        return super(BlackboxFeature, cls).instance(subtype_name, **params)


# entropy of the performance predictor on the prod set
class BlackboxPPEntropy(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('pp_entropy')

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        confs = predictor_output['confidences']
        confs, _ = np.histogram(confs, bins=100, range=[0,1])
        entropy = compute_average_entropy(confs)
        return entropy


# confidence std predicted by performance predictor
class BlackboxPredictedStd(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('pp_std')

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        pp_std = np.std(predictor_output['confidences'])
        return pp_std


# confidence interval width predicted by performance predictor
class BlackboxPredictedBootstrap(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('pp_bootstrap')

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        st = SignificanceTester("average")
        CI = st.confidence_interval(predictor_output['confidences'])
        size = float(CI[2] - CI[1])
        return size


# ratio of base model entropy on the test set vs. prod set
class BlackboxBaseEntropyRatio(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('base_model_entropy_ratio')

    def extract_payload(self, base_predictions: np.ndarray, y_test: np.ndarray):
        payload = {'test_entropy': compute_average_entropy(base_predictions)}
        return payload

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        prod_entropy = compute_average_entropy(base_model_output)
        test_entropy = payload['test_entropy']
        return prod_entropy / test_entropy


# accuracy change predicted by performance predictor
class BlackboxPredictedDrop(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('predicted_accuracy_change')

    def extract_payload(self, base_predictions: np.ndarray, y_test: np.ndarray):
        predicted_labels = self.convert_to_labels( np.argmax(base_predictions, axis=1) )
        accuracy = 100.0 * np.mean(np.where(predicted_labels == y_test, 1, 0))
        return {'test_accuracy': accuracy}

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        predicted_prod_accuracy = 100.0 * np.mean(predictor_output['confidences'])
        predicted_change = predicted_prod_accuracy - payload['test_accuracy']
        return predicted_change


# average uncertainty predicted by performance predictor
class BlackboxPredictedUncertainty(BlackboxFeature):
    def __init__(self, train_labels=None):
        super().__init__(train_labels=train_labels)

    @classmethod
    def name(cls):
        return ('pp_uncertainty')

    def extract_blackbox_feature(self, base_model_output, predictor_output, payload: dict):
        predicted_uncertainty = 100.0 * np.mean(predictor_output['uncertainties'])
        return predicted_uncertainty
