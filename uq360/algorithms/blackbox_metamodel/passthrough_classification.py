from collections import namedtuple

from uq360.algorithms.blackbox_metamodel.predictors.predictor_driver import PredictorDriver
from uq360.algorithms.posthocuq import PostHocUQ


class PassthroughClassificationWrapper(PostHocUQ):

    def __init__(self, base_model=None):
        """
        Simply passes the predicted class confidence of the base/input model as its own prediction. This performance predictor does not have a method to quantify its own uncertainty, so the uncertainty values are zero.
        PostHocUQ model based on the "passthrough" performance predictor (uq360.algorithms.blackbox_metamodel.predictors.core.passthrough.py).
        """
        super().__init__(base_model)
        self.client_model = base_model
        self.performance_predictor = "passthrough"
        self.calib = 'isotonic_regression'
        self.fitted = False
        self.driver = PredictorDriver(self.performance_predictor,
                                      base_model=self.client_model,
                                      pointwise_features=None,
                                      batch_features=None,
                                      blackbox_features=None,
                                      use_whitebox=True,
                                      use_drift_classifier=True,
                                      calibrator=self.calib)

    def fit(self, x_train, y_train, x_test, y_test, test_predicted_probabilities=None):
        self.driver.fit(x_train, y_train, x_test, y_test, test_predicted_probabilities=test_predicted_probabilities)
        self.fitted = True

    def _process_pretrained_model(self, x, y_hat):
        raise NotImplementedError

    def predict(self, x, return_predictions=True, predicted_probabilities=None):
        """
         Generate a base prediction for incoming data x

        :param x: array-like of shape (n_samples, n_features).
                Features vectors of the test points.
        :param return_predictions: data point wise prediction will be returned when this flag is True
        :param predicted_probabilities: when the predictor is instantiated without a base model, predicted_probabilities on x from the pre-trained model should be passed to predict
        :return: namedtuple: A namedtuple that holds

        y_mean: ndarray of shape (n_samples, [n_output_dims])
            Mean of predictive distribution of the test points.
        y_pred: ndarray of shape (n_samples,)
        Predicted labels of the test points.
        y_score: ndarray of shape (n_samples,)
            Confidence score the test points.

         """
        if not self.fitted:
            raise Exception("Untrained Predictor: fit() method needs to be called before predicting.")

        predictions = self.driver.predict(x, predicted_probabilities=predicted_probabilities)

        output = {'predicted_accuracy': predictions['accuracy'], 'uncertainty': predictions['uncertainty']}
        if 'error' in predictions:
            output['error'] = predictions['error']

        if return_predictions:
            output['predictions_per_datapoint'] = predictions['pointwise_confidences']


        Result = namedtuple('res',['y_mean', 'y_pred', 'y_score'])
        res = Result(predictions['accuracy'], [], [predictions['pointwise_confidences']])

        return res
