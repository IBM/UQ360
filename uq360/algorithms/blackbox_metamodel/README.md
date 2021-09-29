# Blackbox metamodel predictors

This directory contains algorithms for predicting the performance of ML models on unlabeled data. 

As an example, most of the algorithms in this directory will report a dictionary that contains the "predicted_accuracy"

```
{'predicted_accuracy': 54.5} 
```
which means the client model's performance on unlabeled data is 54.5.

## Terminology

- "Predictor" is used to predict the performance of the ML model on unlabeled data
- "Base model" / "Client model" represents the (pre-trained) model from the user.
- "prod data" / "unlabeled data" represents data seen by the model in real world. The model is unaware of this data during training
- "Drift scenario" refers to the "drift/bias" in data which leads to a model that performs differently on the test set and production set

### Predictors 

#### Structured Data Predictor
This predictor allows flexible feature and calibrator configurations, and uses a meta-model which is an ensemble of a GBM and a Logistic Regression model. It returns no errorbars (constant zero errorbars) of its own.

#### Short Text Predictor
This is very similar to the structured data predictor but it is fine tuned to handle text data. The meta model used by the predictor is an ensemble of an SVM, GBM, and MLP. Feature vectors can be either raw text or pre-encoded vectors. If raw text is passed and no encoder is specified in the initialization, USE embeddings will be used by default. 

#### Confidence Predictor
This is a simple predictor which bins predictions of the base model based on their highest confidence value, and returns dynamic errorbars determined by the standard deviation of values in each bin.

#### Passthrough Predictor
Simply passes the predicted class confidence of the base/input model as its own prediction. This performance predictor does not have a method to quantify its own uncertainty, so the uncertainty 
values are zero.  

### Usage

```
# load base model


base_model= <sklearn estimator>

x_train = <numpy nd.array>
y_train = <numpy nd.array>

x_test = <numpy nd.array>
y_test = <numpy nd.array>

x_prod = <numpy nd.array> 
```

```
# create an object

# Instantiation example for structured data predictor
from uq360.algorithms.blackbox_metamodel.structured_data_predictor import StructuredDataPredictorWrapper
p = StructuredDataPredictorWrapper(base_model=<sklearn estimator>)

# Instantiation example for short text data predictor
from uq360.algorithms.blackbox_metamodel.short_text_predictor import ShortTextPredictorWrapper
p = ShortTextPredictorWrapper(base_model=<sklearn estimator>)

# fit a predictor (with train and test data)
p.fit(x_train, y_train, x_test, y_test)

# predict performance on x_prod (unlabeled data)
# The return_predictions flag (default=False) controls whether to include the point_wise predictions
prediction = p.predict(x_prod, return_predictions=True)

```
* `StructuredDataPredictorWrapper` / `ShortTextPredictorWrapper` instantiation Arguments:
  * model (sklearn estimators https://scikit-learn.org/stable/developers/develop.html) : (Trained) Client model<sup>1</sup> (optional, but if not included then model confidence vectors must be provided to fit and predict)
  * encoder (TransformerMixin object that implements fit and transform methods): This is an optional parameter and should be set only for text data (when predictor=text_ensemble). If the predictor is "text_ensemble" and if encoder is not set, the default encoding used will be USE Encoding.

* `fit()` arguments:
  * x_train (numpy.ndarray) : Training data features used for base model. For text data, x_train can be an encoded vector or raw text (USE encoding will be used when an encoder is not specified) <sup>2</sup>
  * y_train (numpy.ndarray) : Training data labels (1 dimensional - non-onehot encoded).
  * x_test (numpy.ndarray) : Test data features used for base model evaluation.
  * y_train (numpy.ndarray) : Test data labels (1 dimensional - non-onehot encoded).
  * test_predicted_probabilities (numpy.ndarray) : Confidence vectors for each sample in x_test (not required if base model provided to constructor)

* `predict()` arguments:
  * x_prod (numpy.ndarray) : Feature vectors from production data batch. When predict() is used with text data, encoded vectors can be supplied as input or raw text can also be fed as input. USE encoding will be applied automatically if "encoder" is not set during instantiation.
  * return_uncertainty (boolean) : Default is True.
  * return_predictions (boolean) : Default is False (When this flag is enabled, the API will return 'predictions_per_datapoint' in addition to predicted accuracy and uncertainty)
  * predicted_probabilities (numpy.ndarray) : Confidence vectors for each sample in x_prod (not required if base model provided to constructor)


* Returns : dictionary -- `{predicted_accuracy(%)}` 

Note: code snippets for pass through and confidence predictor

```
from uq360.algorithms.blackbox_metamodel.confidence_predictor import ConfidencePredictor
p = ConfidencePredictor(base_model=<sklearn estimator>)

```

```
from uq360.algorithms.blackbox_metamodel.passthrough_predictor import PassthroughPredictor
p = PassthroughPredictor(base_model=<sklearn estimator>)

```

To learn more about usage, refer to notebooks 

https://github.com/anupamamurthi/UQ360/blob/main/examples/blackbox_metamodel/Short%20text%20predictor%20Demo.ipynb

https://github.com/anupamamurthi/UQ360/blob/main/examples/blackbox_metamodel/Structured%20Data%20Predictor%20(Ensemble).ipynb

