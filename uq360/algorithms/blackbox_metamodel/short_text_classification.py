from collections import namedtuple

import numpy as np

from uq360.algorithms.blackbox_metamodel.predictors.predictor_driver import PredictorDriver
from uq360.algorithms.posthocuq import PostHocUQ
from uq360.utils.utils import UseTransformer
import logging

logger = logging.getLogger(__name__)


class ShortTextClassificationWrapper(PostHocUQ):
    """
    This is very similar to the structured data predictor but it is fine tuned to handle text data. The meta model used by the predictor is an ensemble of an SVM, GBM, and MLP. Feature vectors can be either raw text or pre-encoded vectors. If raw text is passed and no encoder is specified in the initialization, USE embeddings will be used by default.
    PostHocUQ model based on the "text_ensemble" performance predictor (uq360.algorithms.blackbox_metamodel.predictors.core.short_text.py).
    """

    def __init__(self, base_model=None, encoder=None):
        """ Returns an instance of a short text predictor
        :param base_model: scikit learn estimator instance which has the capability of returning confidence (predict_proba). base_model can also be None
        :return: predictor instance
        """
        super().__init__(base_model)

        self.encoder = None
        self.encoder = UseTransformer()
        self.predictor = "text_ensemble"
        calib = 'shift'
        self.driver = PredictorDriver(self.predictor,
                                      base_model=base_model,
                                      pointwise_features=None,
                                      batch_features=None,
                                      blackbox_features=None,
                                      use_whitebox=True,
                                      use_drift_classifier=True,
                                      calibrator=calib)

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
        if x_train.dtype.type in [np.str_, np.object_] or x_test.dtype.type in [np.str_, np.object_]:
            logger.info('Training/Testing data contains raw text.')
            logger.info('Using an encoder.... %s', self.encoder)
            logger.info('Shapes before encoding %s %s', x_train.shape, x_test.shape)
            x_train = self.encoder.transform(X=x_train)
            x_test = self.encoder.transform(X=x_test)
            logger.info('Shapes after encoding %s %s', x_train.shape, x_test.shape)
        else:
            logger.info('Incoming data is already encoded')

        logger.info("Fitting a text ensemble predictor......")

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

        if x.dtype.type in [np.str_, np.object_]:
            print('Incoming data contains raw text.')
            print('Using an encoder.... %s', self.encoder)
            print('Shapes before encoding %s', x.shape)
            x_prod = self.encoder.transform(X=x)
            print('Shapes after encoding %s', x_prod.shape)
            predictions = self.driver.predict(x_prod, predicted_probabilities=predicted_probabilities)
        else:
            print('Incoming data is already encoded')
            predictions = self.driver.predict(x, predicted_probabilities=predicted_probabilities)

        output = {'predicted_accuracy': predictions['accuracy'], 'uncertainty': predictions['uncertainty']}
        if 'error' in predictions:
            output['error'] = predictions['error']

        if return_predictions:
            output['predictions_per_datapoint'] = predictions['pointwise_confidences']


        Result = namedtuple('res',['y_mean', 'y_pred', 'y_score'])
        res = Result(predictions['accuracy'], [], [predictions['pointwise_confidences']])

        return res
