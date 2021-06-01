# UQ360

[![Build Status](https://travis-ci.com/IBM/UQ360.svg?branch=main)](https://travis-ci.com/github/IBM/UQ360)
[![Documentation Status](https://readthedocs.org/projects/uq360/badge/?version=latest)](https://uq360.readthedocs.io/en/latest/?badge=latest)

The Uncertainty Quantification 360 (UQ360) toolkit is an open-source Python package that provides a diverse set of algorithms to quantify uncertainty, as well as capabilities to measure and improve UQ to streamline the development process. We provide a taxonomy and guidance for choosing these capabilities based on the user's needs. Further, UQ360 makes the communication method of UQ an integral part of development choices in an AI lifecycle. Developers can make a user-centered choice by following the psychology-based guidance on communicating UQ estimates, from concise descriptions to detailed visualizations.

The [UQ360 interactive experience](http://uq360.mybluemix.net/) provides a gentle introduction to the concepts and capabilities by walking through an example use case. The [tutorials and example notebooks](./examples) offer a deeper, data scientist-oriented introduction. The complete API is also available.

We have developed the package with extensibility in mind. This library is still in development. We encourage the contribution of your uncertianty estimation algorithms, metrics and applications. To get started as a contributor, please join the #uq360 channel of the [AIF360 Community on Slack](https://aix360.slack.com) by requesting an invitation [here](https://join.slack.com/t/aix360/shared_invite/enQtNzEyOTAwOTk1NzY2LTM1ZTMwM2M4OWQzNjhmNGRiZjg3MmJiYTAzNDU1MTRiYTIyMjFhZTI4ZDUwM2M1MGYyODkwNzQ2OWQzMThlN2Q).

# Supported Uncertainty Evaluation Metrics

The toolbox provides several standard calibration metrics for classification and regression tasks. This includes Expected Calibration Error ([Naeini et al., 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9602)), Brier Score ([Murphy, 1973](https://en.wikipedia.org/wiki/Brier_score)), etc for classification models. Regression metrics include Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW) among others. The toolbox also provides a novel operation-point agnostic approaches for the assessment of prediction uncertainty estimates called the Uncertainty Characteristic Curve (UCC). Several metrics and diagnosis tools such as reliability diagram ([Niculescu-Mizil & Caruana, 2005](https://openreview.net/forum?id=S1V0en-u-B)) and risk-vs-rejection rate curves are provides which also support analysis by sub-groups in the population to study fairness implications of acting on given uncertainty estimates.

# Supported Uncertainty Estimation Algorithms

UQ algorithms can be broadly classified as intrinsic or extrinsic depending on how the uncertainties are obtained from the AI models. Intrinsic methods encompass models that inherently provides an uncertainty estimate along with its predictions. The toolkit includes algorithms such as variational Bayesian neural  networks  (BNNs)  ([Graves,  2011](https://papers.nips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)),  Gaussian  processes  ([Rasmussen  and  Williams,2006](https://mitpress.mit.edu/books/gaussian-processes-machine-learning)), quantile regression ([Koenker and Bassett, 1978](https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf)) and hetero/homo-scedastic neuralnetworks ([Kendall and Gal, 2017](https://papers.nips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)) which are models that fall in this category The toolkit also includes Horseshoe BNNs ([Ghosh et al., 2019](https://www.jmlr.org/papers/v20/19-236.html)) that use sparsity promoting priors and can lead to better-calibrated uncertainties, especially in the small data regime. An Infinitesimal Jackknife (IJ) based algorithm ([Ghosh et al., 2020)](https://papers.nips.cc/paper/2020/hash/636efd4f9aeb5781e9ea815cdd633e52-Abstract.html)), provided in the toolkit, is a perturbation-based approach that perform uncertainty quantification by estimating model parameters under different perturbations of the original data. Crucially, here the estimation only requires the model to be trained once on the unperturbed dataset. For models that do not have an inherent notion of uncertainty built into them, extrinsic methods are employed to extract uncertainties post-hoc. The toolkit provides meta-models ([Chen et al., 2019](http://proceedings.mlr.press/v89/chen19c.html))that can be been used to successfully generate reliable confidence measures (in classification), prediction intervals (in regression), and to predict performance metrics such as accuracy on unseen and unlabeled data. For pre-trained models that captures uncertainties to some degree, the toolbox provides extrinsic algorithms that can improve the uncertainty estimation quality. This includes isotonic regression ([Zadrozny and Elkan, 2001](https://cseweb.ucsd.edu/~elkan/calibrated.pdf)), Platt-scaling ([Platt, 1999](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639&g)),  auxiliary interval predictors ([Thiagarajan et al., 2020](https://ojs.aaai.org//index.php/AAAI/article/view/6062)), and UCC-Recalibration. 

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
[uq360/datasets/data/README.md](uq360/datasets/data/README.md).

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

## Acknowledgements

UQ360 is built with the help of several open source packages. All of these are listed in setup.py and some of these include: 
* scikit-learn https://scikit-learn.org/stable/about.html
* Pytorch https://github.com/pytorch/pytorch
* Botorch https://github.com/pytorch/botorch

## License Information

Please view both the [LICENSE](./LICENSE) file present in the root directory for license information.
