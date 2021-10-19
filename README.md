# UQ360

[![Build Status](https://travis-ci.com/IBM/UQ360.svg?branch=main)](https://travis-ci.com/github/IBM/UQ360)
[![Documentation Status](https://readthedocs.org/projects/uq360/badge/?version=latest)](https://uq360.readthedocs.io/en/latest/?badge=latest)

The Uncertainty Quantification 360 (UQ360) is an open-source toolkit with a Python package to provide data science 
practitioners and developers access to state-of-the-art algorithms, to streamline the process of estimating, evaluating,
improving, and communicating uncertainty of machine learning models as common practices for AI transparency.
The [UQ360 interactive experience](http://uq360.mybluemix.net/) provides a gentle introduction to the concepts and 
capabilities by walking through an example use case. The [tutorials and example notebooks](./examples) offer a deeper,
data scientist-oriented introduction. The [complete API](https://uq360.readthedocs.io/) is also available.

We have developed the package with extensibility in mind. This library is still in development. We encourage the 
contribution of your uncertainty estimation algorithms, metrics and applications. 
To get started as a contributor, please join the #uq360-users or #uq360-developers channel of 
the [AIF360 Community on Slack](https://aif360.slack.com) by requesting an 
invitation [here](https://join.slack.com/t/aif360/shared_invite/zt-5hfvuafo-X0~g6tgJQ~7tIAT~S294TQ).

![alt text](https://uq360.mybluemix.net/imgs/flowchart.png "UQ Pipeline")

# Resources

- [Introduction](https://uq360.mybluemix.net/overview) to Uncertainty Quantification 360.
- [Demo](https://uq360.mybluemix.net/demo/0) House Price Prediction Model.
- List of [Algorithms](https://uq360.readthedocs.io/en/latest/algorithms.html) supported.
- List of [Metrics](https://uq360.readthedocs.io/en/latest/metrics.html) supported.
- [Guidance](https://uq360.mybluemix.net/resources/guidance) on Choosing UQ Algorithms and Metrics.
- [Guidance](https://uq360.mybluemix.net/resources/communication) on Communicating Uncertainty.
- [Glossary](https://uq360.mybluemix.net/resources/glossary) of UQ Terms.
- Read our [papers](https://uq360.mybluemix.net/resources/papers).
- Complete list of [tutorials](https://github.com/IBM/UQ360/tree/main/examples).
- Join the Slack [Community](https://uq360.mybluemix.net/community).

# Example Use-cases

### Meta-models
Use of meta-models to augment sklearn's gradient boosted regressor with prediction interval. See detailed example 
[here](https://github.com/IBM/UQ360/blob/main/examples/blackbox_metamodel/demo_blackbox_metamodel_regression.ipynb).

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from uq360.algorithms.blackbox_metamodel import MetamodelRegression

# Create train, calibration and test splits.
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train, X_calibration, y_train, y_calibration = train_test_split(X_train, y_train, random_state=0)

# Train the base model that provides the mean estimates.
gbr_reg = GradientBoostingRegressor(random_state=0)
gbr_reg.fit(X_train, y_train)

# Train the meta-model that can augment the mean prediction with prediction intervals.
uq_model = MetamodelRegression(base_model=gbr_reg)
uq_model.fit(X_calibration, y_calibration, base_is_prefitted=True)

# Obtain mean estimates and prediction interval on the test data.
y_hat, y_hat_lb, y_hat_ub = uq_model.predict(X_test)
```

### UQ360 metrics for model selection
The prediction interval coverage probability score (PICP) score is used here 
as the metric to select the model through cross-validation. See detailed example 
[here](https://github.com/IBM/UQ360/blob/main/examples/autoai/demo_autoai.ipynb).

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from uq360.utils.misc import make_sklearn_compatible_scorer
from uq360.algorithms.quantile_regression import QuantileRegression

# Create a sklearn scorer using UQ360 PICP metric.
sklearn_picp = make_sklearn_compatible_scorer(
    task_type="regression",
    metric="picp", greater_is_better=True)

# Hyper-parameters configuration using GridSearchCV.
base_config = {"alpha":0.95, "n_estimators":20, "max_depth": 3, 
               "learning_rate": 0.01, "min_samples_leaf": 10,
               "min_samples_split": 10}
configs  = {"config": []}
for num_estimators in [1, 2, 5, 10, 20, 30, 40, 50]:
    config = base_config.copy()
    config["n_estimators"] = num_estimators
    configs["config"].append(config)

# Create train test split.
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initialize QuantileRegression UQ360 model and wrap it in GridSearchCV with PICP as the scoring function.
uq_model = GridSearchCV(
    QuantileRegression(config=base_config), configs, scoring=sklearn_picp)

# Fit the model on the training set.
uq_model.fit(X_train, y_train)

# Obtain the prediction intervals for the test set.
y_hat, y_hat_lb, y_hat_ub = uq_model.predict(X_test)
```

## Setup

Supported Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.7  |
| Ubuntu  | 3.7  |
| Windows | 3.7  |


### (Optional) Create a virtual environment

A virtual environment manager is strongly recommended to ensure dependencies may be installed safely. If you have trouble installing the toolkit, try this first.

#### Conda

Conda is recommended for all configurations though Virtualenv is generally
interchangeable for our purposes. Miniconda is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) and can be installed from
[here](https://conda.io/miniconda.html) if you do not already have it.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name uq360 python=3.7
conda activate uq360
```

The shell should now look like `(uq360) $`. To deactivate the environment, run:

```bash
(uq360)$ conda deactivate
```

The prompt will return back to `$ ` or `(base)$`.

Note: Older versions of conda may use `source activate uq360` and `source
deactivate` (`activate uq360` and `deactivate` on Windows).


### Installation

Clone the latest version of this repository:

```bash
(uq360)$ git clone https://github.ibm.com/UQ360/UQ360
```

If you'd like to run the examples and tutorial notebooks, download the datasets now and place them in
their respective folders as described in
[uq360/data/README.md](uq360/data/README.md).

Then, navigate to the root directory of the project which contains `setup.py` file and run:

```bash
(uq360)$ pip install -e .
```

## PIP Installation of Uncertainty Quantification 360

If you would like to quickly start using the UQ360 toolkit without cloning this repository, then you can install the [uq360 pypi package](https://pypi.org/project/uq360/) as follows. 

```bash
(your environment)$ pip install uq360
```

If you follow this approach, you may need to download the notebooks in the [examples](./examples) folder separately.

# Using UQ360

The `examples` directory contains a diverse collection of jupyter notebooks that use UQ360 in various ways. Both examples and tutorial notebooks illustrate working code using the toolkit. Tutorials provide additional discussion that walks the user through the various steps of the notebook. See the details about tutorials and examples [here](examples/README.md).

## Citing UQ360

A technical description of UQ360 is available in this
[paper](https://arxiv.org/abs/2106.01410). Below is the bibtex entry for this
paper.

```
@misc{uq360-june-2021,
      title={Uncertainty Quantification 360: A Holistic Toolkit for Quantifying 
      and Communicating the Uncertainty of AI}, 
      author={Soumya Ghosh and Q. Vera Liao and Karthikeyan Natesan Ramamurthy 
      and Jiri Navratil and Prasanna Sattigeri 
      and Kush R. Varshney and Yunfeng Zhang},
      year={2021},
      eprint={2106.01410},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Acknowledgements

UQ360 is built with the help of several open source packages. All of these are listed in setup.py and some of these include: 
* scikit-learn https://scikit-learn.org/stable/about.html
* Pytorch https://github.com/pytorch/pytorch
* Botorch https://github.com/pytorch/botorch

## License Information

Please view both the [LICENSE](./LICENSE) file present in the root directory for license information.
