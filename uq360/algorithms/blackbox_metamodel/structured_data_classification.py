from collections import namedtuple

from uq360.algorithms.blackbox_metamodel.predictors.predictor_driver import PredictorDriver
from uq360.algorithms.posthocuq import PostHocUQ


class StructuredDataClassificationWrapper(PostHocUQ):
    """
    This predictor allows flexible feature and calibrator configurations, and uses a meta-model which is an ensemble of a GBM and a Logistic Regression model. It returns no errorbars (constant zero errorbars) of its own.
    PostHocUQ model based on the "structured_data" performance predictor
            (uq360.algorithms.blackbox_metamodel.predictors.core.structured_data.py).
    """
    def __init__(self, base_model=None):
        """ Returns an instance of a structured data predictor

        :param base_model: scikit learn estimator instance which has the capability of returning confidence (predict_proba). base_model can also be None
        :return: predictor instance
        """
        super().__init__(base_model)
        self.client_model = base_model
        self.performance_predictor = "structured_data"
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
        """
                       Fit base and meta models.

                       :param x_train: Features vectors of the training data.
                       :param y_train: Labels of the training data
                       :param x_test: Features vectors of the test data.
                       :param y_test: Labels of the test data
                       :param test_predicted_probabilities: predicted probabilities on test data should be passed if the predictor is not instantiated with a base model
                       :return: self
                """
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