This directory contains a diverse collection of jupyter notebooks that use UQ360 toolkit in various ways. Both tutorials and examples illustrate working code using the toolkit. Tutorials provide additional discussion that walks the user through the various steps of the notebook.

## Tutorials

- [Calibrated Housing Price Prediction](./tutorials/tutorial-synthetic_housing_dataset.ipynb)<br/>Illustrates using the housing price prediction task the consumption of uncertainties by two personas: the decision maker and the model developer. This tutorial covers the basic concepts of uncertainty quantification and evaluation in a simple regression setting.

- [Selective Classification on Adult Income Dataset](./tutorials/tutorial-adult_income_dataset.ipynb
)<br/> Shows the usage of uncertainty information to help end users and developers compare the model's performance with male v.s. female customers. This tutorial covers the basic concepts of uncertainty quantification and evaluation in the selective classification setting.

## Examples

Below is a list of additional notebooks that demonstrate the use of UQ360:

[demo_actively_learned_model_methods_comparisons.ipynb](./actively_learned_model/demo_actively_learned_model_methods_comparisons.ipynb): illustrates the use of active learning to train a surrogate model of Partial Differential Equations (PDE) and shows the benefit compared to random sampling.

[demo_auxiliary_interval_predictor.ipynb](./auxiliary_interval_predictor/demo_auxiliary_interval_predictor.ipynb): demonstrates the use of Auxiliary Interval Predictors for calibration aware regression.

[demo_blackbox_metamodel_regression.ipynb](./blackbox_metamodel/demo_blackbox_metamodel_regression.ipynb): demonstrates the use of Blackbox MetaModel for post-hoc extraction of uncertainty from pre-trained regression models.

[demo_blackbox_metamodel_classification.ipynb](./blackbox_metamodel/demo_blackbox_metamodel_classification.ipynb): demonstrates the use of Blackbox MetaModel for post-hoc extraction of uncertainty from pre-trained classification models.

[demo_bnn_classification.ipynb](./bnn_classification/demo_bnn_classification.ipynb
): demonstrates the use of BNNs and uncertainty decomposition for selective classificaiton on UCI Adult Income Dataset.

[demo_gp_regression_meps_dataset.ipynb](./gp_regression/demo_gp_regression_meps_dataset.ipynb
): demonstrates the use of Gaussian Process regression on MEPS dataset for the healthcare utlization prediction task.

[demo_heteroscedastic_regression.ipynb](./heteroscedastic_regression/demo_heteroscedastic_regression.ipynb
): demonstrates the use of regression with heteroscedastic noise.

[demo_ensemble_heteroscedastic_regression.ipynb](./ensemble_heteroscedastic_regression/demo_ensemble_heteroscedastic_regression.ipynb
): demonstrates the use of an ensemble of regression with heteroscedastic noise.

[demo_infinitesimal_jackknife.ipynb](./infinitesimal_jackknife/demo_infinitesimal_jackknife.ipynb): illustrates the infinitesimal jackknife (IJ) for logistic regression.

[demo_quantile_regression.ipynb](./quantile_regression/demo_quantile_regression.ipynb
): demostrates the use of regression with quantile loss.

[demo_structured_infinitesimal_jackknife.ipynb](./infinitesimal_jackknife/demo_structured_infinitesimal_jackknife.ipynb): demonstrates structured infinitesimal jackknife (IJ) approximations for cases where data are not independent across folds of the jackknife.

[demo_ucc_class.ipynb](./ucc_metric/demo_ucc_class.ipynb): demonstrates the basic usage of the Uncertainty Characteristics Curve (UCC).
