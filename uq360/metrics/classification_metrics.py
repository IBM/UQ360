import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


def entropy_based_uncertainty_decomposition(y_prob_samples):
    """ Entropy based decomposition [2]_ of predictive uncertainty into aleatoric and epistemic components.

    References:
        .. [2] Depeweg, S., Hernandez-Lobato, J. M., Doshi-Velez, F., & Udluft, S. (2018, July). Decomposition of
            uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. In International Conference
            on Machine Learning (pp. 1184-1193). PMLR.

    Args:
        y_prob_samples: ndarray of shape (mc_samples, n_samples, n_classes)
            Samples from the predictive distribution. Here mc_samples stands for the number of Monte-Carlo samples,
            n_samples is the number of data points and n_classes is the number of classes.

    Returns:
        tuple:
            - total_uncertainty: entropy of the predictive distribution.
            - aleatoric_uncertainty: aleatoric component of the total_uncertainty.
            - epistemic_uncertainty: epistemic component of the total_uncertainty.

    """
    prob_mean = np.mean(y_prob_samples, 0)

    total_uncertainty = entropy(prob_mean, axis=1)
    aleatoric_uncertainty = np.mean(
        np.concatenate([entropy(y_prob, axis=1).reshape(-1, 1) for y_prob in y_prob_samples], axis=1),
        axis=1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def multiclass_brier_score(y_true, y_prob):
    """Brier score for multi-class.

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.

    Returns:
        float: Brier score.

    """
    assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"

    y_target = np.zeros_like(y_prob)
    y_target[np.arange(y_true.size), y_true] = 1.0
    return np.mean(np.sum((y_target - y_prob) ** 2, axis=1))


def area_under_risk_rejection_rate_curve(y_true, y_prob, y_pred=None, selection_scores=None, risk_func=accuracy_score,
                                         attributes=None, num_bins=10, subgroup_ids=None,
                                         return_counts=False):
    """ Computes risk vs rejection rate curve and the area under this curve. Similar to risk-coverage curves [3]_ where
    coverage instead of rejection rate is used.

    References:
        .. [3] Franc, Vojtech, and Daniel Prusa. "On discriminative learning of prediction uncertainty."
         In International Conference on Machine Learning, pp. 1963-1971. 2019.

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        y_pred: array-like of shape (n_samples,)
            predicted labels.
        selection_scores: scores corresponding to certainty in the predicted labels.
        risk_func: risk function under consideration.
        attributes: (optional) if risk function is a fairness metric also pass the protected attribute name.
        num_bins: number of bins.
        subgroup_ids: (optional) selectively compute risk on a subgroup of the samples specified by subgroup_ids.
        return_counts: set to True to return counts also.

    Returns:
        float or tuple:
            - aurrrc (float): area under risk rejection rate curve.
            - rejection_rates (list): rejection rates for each bin (returned only if return_counts is True).
            - selection_thresholds (list): selection threshold for each bin (returned only if return_counts is True).
            - risks (list): risk in each bin (returned only if return_counts is True).

    """

    if selection_scores is None:
        assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"
        selection_scores = y_prob[np.arange(y_prob.shape[0]), np.argmax(y_prob, axis=1)]

    if y_pred is None:
        assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"
        y_pred = np.argmax(y_prob, axis=1)

    order = np.argsort(selection_scores)[::-1]

    rejection_rates = []
    selection_thresholds = []
    risks = []
    for bin_id in range(num_bins):
        samples_in_bin = len(y_true) // num_bins
        selection_threshold = selection_scores[order[samples_in_bin * (bin_id+1)-1]]
        selection_thresholds.append(selection_threshold)
        ids = selection_scores >= selection_threshold
        if sum(ids) > 0:
            if attributes is None:
                if isinstance(y_true, pd.Series):
                    y_true_numpy = y_true.values
                else:
                    y_true_numpy = y_true
                if subgroup_ids is None:
                    risk_value = 1.0 - risk_func(y_true_numpy[ids], y_pred[ids])
                else:
                    if sum(subgroup_ids & ids) > 0:
                        risk_value = 1.0 - risk_func(y_true_numpy[subgroup_ids & ids], y_pred[subgroup_ids & ids])
                    else:
                        risk_value = 0.0
            else:
                risk_value = risk_func(y_true.iloc[ids], y_pred[ids], prot_attr=attributes)
        else:
            risk_value = 0.0
        risks.append(risk_value)
        rejection_rates.append(1.0 - 1.0 * sum(ids) / len(y_true))

    aurrrc = np.nanmean(risks)

    if not return_counts:
        return aurrrc
    else:
        return aurrrc, rejection_rates, selection_thresholds, risks


def expected_calibration_error(y_true, y_prob, y_pred=None, num_bins=10, return_counts=False):
    """ Computes the reliability curve and the  expected calibration error [1]_ .

    References:
        .. [1] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger; Proceedings of the 34th International Conference
         on Machine Learning, PMLR 70:1321-1330, 2017.

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        y_pred: array-like of shape (n_samples,)
            predicted labels.
        num_bins: number of bins.
        return_counts: set to True to return counts also.

    Returns:
        float or tuple:
            - ece (float): expected calibration error.
            - confidences_in_bins: average confidence in each bin (returned only if return_counts is True).
            - accuracies_in_bins: accuracy in each bin (returned only if return_counts is True).
            - frac_samples_in_bins: fraction of samples in each bin (returned only if return_counts is True).

    """

    assert len(y_prob.shape) > 1, "y_prob should be array-like of shape (n_samples, n_classes)"
    num_samples, num_classes = y_prob.shape
    top_scores = np.max(y_prob, axis=1)

    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    if num_classes == 2:
        bins_edges = np.histogram_bin_edges([], bins=num_bins, range=(0.5, 1.0))
    else:
        bins_edges = np.histogram_bin_edges([], bins=num_bins, range=(0.0, 1.0))

    non_boundary_bin_edges = bins_edges[1:-1]
    bin_centers = (bins_edges[1:] + bins_edges[:-1])/2

    sample_bin_ids = np.digitize(top_scores, non_boundary_bin_edges)

    num_samples_in_bins = np.zeros(num_bins)
    accuracies_in_bins = np.zeros(num_bins)
    confidences_in_bins = np.zeros(num_bins)

    for bin in range(num_bins):
        num_samples_in_bins[bin] = len(y_pred[sample_bin_ids == bin])
        if num_samples_in_bins[bin] > 0:
            accuracies_in_bins[bin] = np.sum(y_true[sample_bin_ids == bin] == y_pred[sample_bin_ids == bin]) / num_samples_in_bins[bin]
            confidences_in_bins[bin] = np.sum(top_scores[sample_bin_ids == bin]) / num_samples_in_bins[bin]

    ece = np.sum(
        num_samples_in_bins * np.abs(accuracies_in_bins - confidences_in_bins) / num_samples
    )
    frac_samples_in_bins = num_samples_in_bins / num_samples

    if not return_counts:
        return ece
    else:
        return ece, confidences_in_bins, accuracies_in_bins, frac_samples_in_bins, bin_centers


def compute_classification_metrics(y_true, y_prob, option='all'):
    """
    Computes the metrics specified in the option which can be string or a list of strings. Default option `all` computes
    the [aurrrc, ece, auroc, nll, brier, accuracy] metrics.

    Args:
        y_true: array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        option: string or list of string contained the name of the metrics to be computed.

    Returns:
        dict: a dictionary containing the computed metrics.
    """
    results = {}
    if not isinstance(option, list):
        if option == "all":
            option_list = ["aurrrc", "ece", "auroc", "nll", "brier", "accuracy"]
        else:
            option_list = [option]
    else:
        option_list = option

    if "aurrrc" in option_list:
        results["aurrrc"] = area_under_risk_rejection_rate_curve(y_true=y_true, y_prob=y_prob)
    if "ece" in option_list:
        results["ece"] = expected_calibration_error(y_true=y_true, y_prob=y_prob)
    if "auroc" in option_list:
        results["auroc"] = roc_auc_score(y_true=y_true, y_score=y_prob, multi_class='ovr')
    if "nll" in option_list:
        results["nll"] = log_loss(y_true=y_true, y_pred=y_prob)
    if "brier" in option_list:
        results["brier"] = multiclass_brier_score(y_true=y_true, y_prob=y_prob)
    if "accuracy" in option_list:
        results["accuracy"] = accuracy_score(y_true=y_true, y_pred=np.argmax(y_prob, axis=1))

    return results


def plot_reliability_diagram(y_true, y_prob, y_pred, plot_label=[""], num_bins=10):
    """
    Plots the reliability diagram showing the calibration error for different confidence scores. Multiple curves
    can be plot by passing data as lists.

    Args:
        y_true: array-like or or a list of array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like or or a list of array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        y_pred: array-like or or a list of array-like of shape (n_samples,)
            predicted labels.
        plot_label: (optional) list of names identifying each curve.
        num_bins: number of bins.

    Returns:
        tuple:
            - ece_list: ece: list containing expected calibration error for each curve.
            - accuracies_in_bins_list: list containing binned average accuracies for each curve.
            - frac_samples_in_bins_list: list containing binned sample frequencies for each curve.
            - confidences_in_bins_list: list containing binned average confidence for each curve.
    """
    import matplotlib.pyplot as plt

    if not isinstance(y_true, list):
        y_true, y_prob, y_pred = [y_true], [y_prob], [y_pred]
    if len(plot_label) != len(y_true):
        raise ValueError('y_true and plot_label should be of same length.')

    ece_list = []
    accuracies_in_bins_list = []
    frac_samples_in_bins_list = []
    confidences_in_bins_list = []

    for idx in range(len(plot_label)):
        ece, confidences_in_bins, accuracies_in_bins, frac_samples_in_bins, bins = expected_calibration_error(y_true[idx],
                                                                                                  y_prob[idx],
                                                                                                  y_pred[idx],
                                                                                                  num_bins=num_bins,
                                                                                                  return_counts=True)
        ece_list.append(ece)
        accuracies_in_bins_list.append(accuracies_in_bins)
        frac_samples_in_bins_list.append(frac_samples_in_bins)
        confidences_in_bins_list.append(confidences_in_bins)

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for idx in range(len(plot_label)):
        plt.plot(bins, frac_samples_in_bins_list[idx], 'o-', label=plot_label[idx])
    plt.title("Confidence Histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Fraction of Samples")
    plt.grid()
    plt.ylim([0.0, 1.0])
    plt.legend()

    plt.subplot(1, 2, 2)
    for idx in range(len(plot_label)):
        plt.plot(bins, accuracies_in_bins_list[idx], 'o-',
                 label="{} ECE = {:.2f}".format(plot_label[idx], ece_list[idx]))
    plt.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50), 'b.', label="Perfect Calibration")
    plt.title("Reliability Plot")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.show()

    return ece_list, accuracies_in_bins_list, frac_samples_in_bins_list, confidences_in_bins_list


def plot_risk_vs_rejection_rate(y_true, y_prob, y_pred, selection_scores=None, plot_label=[""], risk_func=None,
                                attributes=None, num_bins=10, subgroup_ids=None):
    """
    Plots the risk vs rejection rate curve showing the risk for different rejection rates. Multiple curves
    can be plot by passing data as lists.

    Args:
        y_true: array-like or or a list of array-like of shape (n_samples,)
            ground truth labels.
        y_prob: array-like or or a list of array-like of shape (n_samples, n_classes).
            Probability scores from the base model.
        y_pred: array-like or or a list of array-like of shape (n_samples,)
            predicted labels.
        selection_scores: ndarray or a list of ndarray containing scores corresponding to certainty in the predicted labels.
        risk_func: risk function under consideration.
        attributes: (optional) if risk function is a fairness metric also pass the protected attribute name.
        num_bins: number of bins.
        subgroup_ids: (optional) ndarray or a list of ndarray containing subgroup_ids to selectively compute risk on a
            subgroup of the samples specified by subgroup_ids.

    Returns:
        tuple:
            - aurrrc_list: list containing the area under risk rejection rate curves.
            - rejection_rate_list: list containing the binned rejection rates.
            - selection_thresholds_list: list containing the binned selection thresholds.
            - risk_list: list containing the binned risks.
    """
    import matplotlib.pyplot as plt

    if not isinstance(y_true, list):
        y_true, y_prob, y_pred, selection_scores, subgroup_ids = [y_true], [y_prob], [y_pred], [selection_scores], [subgroup_ids]
    if len(plot_label) != len(y_true):
        raise ValueError('y_true and plot_label should be of same length.')

    aurrrc_list = []
    rejection_rate_list = []
    risk_list = []
    selection_thresholds_list = []

    for idx in range(len(plot_label)):
        aursrc, rejection_rates, selection_thresholds, risks = area_under_risk_rejection_rate_curve(
            y_true[idx],
            y_prob[idx],
            y_pred[idx],
            selection_scores=selection_scores[idx],
            risk_func=risk_func,
            attributes=attributes,
            num_bins=num_bins,
            subgroup_ids=subgroup_ids[idx],
            return_counts=True
        )

        aurrrc_list.append(aursrc)
        rejection_rate_list.append(rejection_rates)
        risk_list.append(risks)
        selection_thresholds_list.append(selection_thresholds)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for idx in range(len(plot_label)):
        plt.plot(rejection_rate_list[idx], risk_list[idx], label="{} AURRRC={:.5f}".format(plot_label[idx], aurrrc_list[idx]))

    plt.legend(loc="best")
    plt.xlabel("Rejection Rate")
    if risk_func is None:
        ylabel = "Prediction Error Rate"
    else:
        if 'accuracy' in risk_func.__name__:
            ylabel = "1.0 - " + risk_func.__name__
        else:
            ylabel = risk_func.__name__

    plt.ylabel(ylabel)
    plt.title("Risk vs Rejection Rate Plot")
    plt.grid()

    plt.subplot(1, 2, 2)
    for idx in range(len(plot_label)):
        plt.plot(selection_thresholds_list[idx], risk_list[idx], label="{}".format(plot_label[idx]))

    plt.legend(loc="best")
    plt.xlabel("Selection Threshold")
    if risk_func is None:
        ylabel = "Prediction Error Rate"
    else:
        if 'accuracy' in risk_func.__name__:
            ylabel = "1.0 - " + risk_func.__name__
        else:
            ylabel = risk_func.__name__

    plt.ylabel(ylabel)
    plt.title("Risk vs Selection Threshold Plot")
    plt.grid()

    plt.show()

    return aurrrc_list, rejection_rate_list, selection_thresholds_list, risk_list
