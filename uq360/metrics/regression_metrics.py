import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score

from ..utils.misc import fitted_ucc_w_nullref


def picp(y_true, y_lower, y_upper):
    """
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.

    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_lower, y_upper):
    """
    Mean Prediction Interval Width (MPIW). Computes the average width of the the prediction intervals. Measures the
    sharpness of intervals.

    Args:
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the average width the prediction interval across samples.
    """
    return np.mean(np.abs(y_lower - y_upper))


def auucc_gain(y_true, y_mean, y_lower, y_upper):
    """ Computes the Area Under the Uncertainty Characteristics Curve (AUUCC) gain wrt to a null reference
    with constant band.

    Args:
        y_true: Ground truth
        y_mean: predicted mean
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: AUUCC gain

    """
    u = fitted_ucc_w_nullref(y_true, y_mean, y_lower, y_upper)
    auucc = u.get_AUUCC()
    assert(isinstance(auucc, list) and len(auucc) == 2), "Failed to calculate auucc gain"
    assert (not np.isclose(auucc[1], 0.)), "Failed to calculate auucc gain"
    auucc_gain = (auucc[1]-auucc[0])/auucc[0]
    return auucc_gain


def negative_log_likelihood_Gaussian(y_true, y_mean, y_lower, y_upper):
    """ Computes Gaussian negative_log_likelihood assuming symmetric band around the mean.

    Args:
        y_true: Ground truth
        y_mean: predicted mean
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: nll

    """
    y_std = (y_upper - y_lower) / 4.0
    nll = np.mean(-norm.logpdf(y_true.squeeze(), loc=y_mean.squeeze(), scale=y_std.squeeze()))
    return nll


def compute_regression_metrics(y_true, y_mean, y_lower, y_upper, option="all", nll_fn=None):
    """
    Computes the metrics specified in the option which can be string or a list of strings. Default option `all` computes
    the ["rmse", "nll", "auucc_gain", "picp", "mpiw", "r2"] metrics.

    Args:
        y_true: Ground truth
        y_mean: predicted mean
        y_lower: predicted lower bound
        y_upper: predicted upper bound
        option: string or list of string contained the name of the metrics to be computed.
        nll_fn: function that evaluates NLL, if None, then computes Gaussian NLL using y_mean and y_lower.

    Returns:
        dict: dictionary containing the computed metrics.
    """

    assert y_true.shape == y_mean.shape, "y_true shape: {}, y_mean shape: {}".format(y_true.shape, y_mean.shape)
    assert y_true.shape == y_lower.shape, "y_true shape: {}, y_mean shape: {}".format(y_true.shape, y_lower.shape)
    assert y_true.shape == y_upper.shape, "y_true shape: {}, y_mean shape: {}".format(y_true.shape, y_upper.shape)

    results = {}
    if not isinstance(option, list):
        if option == "all":
            option_list = ["rmse", "nll", "auucc_gain", "picp", "mpiw", "r2"]
        else:
            option_list = [option]
    else:
        option_list = option

    if "rmse" in option_list:
        results["rmse"] = mean_squared_error(y_true, y_mean, squared=False)
    if "nll" in option_list:
        if nll_fn is None:
            nll = negative_log_likelihood_Gaussian(y_true, y_mean, y_lower, y_upper)
            results["nll"] = nll
        else:
            results["nll"] = np.mean(nll_fn(y_true))
    if "auucc_gain" in option_list:
        gain = auucc_gain(y_true, y_mean, y_lower, y_upper)
        results["auucc_gain"] = gain
    if "picp" in option_list:
        results["picp"] = picp(y_true, y_lower, y_upper)
    if "mpiw" in option_list:
        results["mpiw"] = mpiw(y_lower, y_upper)
    if "r2" in option_list:
        results["r2"] = r2_score(y_true, y_mean)

    return results


def _check_not_tuple_of_2_elements(obj, obj_name='obj'):
    """Check object is not tuple or does not have 2 elements."""
    if not isinstance(obj, tuple) or len(obj) != 2:
        raise TypeError('%s must be a tuple of 2 elements.' % obj_name)


def plot_uncertainty_distribution(dist, show_quantile_dots=False, qd_sample=20, qd_bins=7,
    ax=None, figsize=None, dpi=None,
    title='Predicted Distribution', xlims=None, xlabel='Prediction', ylabel='Density', **kwargs):
    """
    Plot the uncertainty distribution for a single distribution.

    Args:
        dist: scipy.stats._continuous_distns.
            A scipy distribution object.
        show_quantile_dots: boolean.
            Whether to show quantil dots on top of the density plot.
        qd_sample: int.
            Number of dots for the quantile dot plot.
        qd_bins: int.
            Number of bins for the quantile dot plot.
        ax: matplotlib.axes.Axes or None, optional (default=None).
            Target axes instance. If None, new figure and axes will be created.
        figsize: tuple of 2 elements or None, optional (default=None).
            Figure size.
        dpi : int or None, optional (default=None).
            Resolution of the figure.
        title : string or None, optional (default=Prediction Distribution)
            Axes title.
            If None, title is disabled.
        xlims : tuple of 2 elements or None, optional (default=None). Tuple passed to ``ax.xlim()``.
        xlabel : string or None, optional (default=Prediction)
            X-axis title label.
            If None, title is disabled.
        ylabel : string or None, optional (default=Density)
            Y-axis title label.
            If None, title is disabled.

    Returns:
        matplotlib.axes.Axes: ax : The plot with prediction distribution.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    ax.plot(x, dist.pdf(x), **kwargs)

    if show_quantile_dots:
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection
        import matplotlib.ticker as ticker

        data = dist.rvs(size=10000)
        p_less_than_x = np.linspace(1 / qd_sample / 2, 1 - (1 / qd_sample / 2), qd_sample)
        x_ = np.percentile(data, p_less_than_x * 100)  # Inverce CDF (ppf)
        # Create bins
        hist = np.histogram(x_, bins=qd_bins)
        bins, edges = hist
        radius = (edges[1] - edges[0]) / 2

        ax2 = ax.twinx()
        patches = []
        max_y = 0
        for i in range(qd_bins):
            x_bin = (edges[i + 1] + edges[i]) / 2
            y_bins = [(i + 1) * (radius * 2) for i in range(bins[i])]

            max_y = max(y_bins) if max(y_bins) > max_y else max_y

            for _, y_bin in enumerate(y_bins):
                circle = Circle((x_bin, y_bin), radius)
                patches.append(circle)

        p = PatchCollection(patches, alpha=0.4)
        ax2.add_collection(p)

        # Axis tweek
        y_scale = (max_y + radius) / max(dist.pdf(x))
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x_ / y_scale))
        ax2.yaxis.set_major_formatter(ticks_y)
        ax2.set_yticklabels([])
        if xlims is not None:
            ax2.set_xlim(left=xlims[0], right=xlims[1])
        else:
            ax2.set_xlim([min(x_) - radius, max(x) + radius])
        ax2.set_ylim([0, max_y + radius])
        ax2.set_aspect(1)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    return ax


def plot_picp_by_feature(x_test, y_test, y_test_pred_lower_total, y_test_pred_upper_total, num_bins=10,
                        ax=None, figsize=None, dpi=None, xlims=None, ylims=None, xscale="linear",
                                title=None, xlabel=None, ylabel=None):
    """
    Plot how prediction uncertainty varies across the entire range of a feature.

    Args:
        x_test: One dimensional ndarray.
            Feature column of the test dataset.
        y_test: One dimensional ndarray.
            Ground truth label of the test dataset.
        y_test_pred_lower_total: One dimensional ndarray.
            Lower bound of the total uncertainty range.
        y_test_pred_upper_total: One dimensional ndarray.
            Upper bound of the total uncertainty range.
        num_bins: int.
            Number of bins used to discritize x_test into equal-sample-sized bins.
        ax: matplotlib.axes.Axes or None, optional (default=None). Target axes instance. If None, new figure and axes will be created.
        figsize: tuple of 2 elements or None, optional (default=None). Figure size.
        dpi : int or None, optional (default=None). Resolution of the figure.
        xlims : tuple of 2 elements or None, optional (default=None). Tuple passed to ``ax.xlim()``.
        ylims: tuple of 2 elements or None, optional (default=None). Tuple passed to ``ax.ylim()``.
        xscale: Passed to ``ax.set_xscale()``.
        title : string or None, optional
            Axes title.
            If None, title is disabled.
        xlabel : string or None, optional
            X-axis title label.
            If None, title is disabled.
        ylabel : string or None, optional
            Y-axis title label.
            If None, title is disabled.

    Returns:
        matplotlib.axes.Axes: ax : The plot with PICP scores binned by a feature.

    """
    from scipy.stats.mstats import mquantiles
    import matplotlib.pyplot as plt

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    x_uniques_sorted = np.sort(np.unique(x_test))

    num_unique = len(x_uniques_sorted)
    sample_bin_ids = np.searchsorted(x_uniques_sorted, x_test)
    if len(x_uniques_sorted) > 10:  # bin the values
            q_bins = mquantiles(x_test, np.histogram_bin_edges([], bins=num_bins-1, range=(0.0, 1.0))[1:])
            q_sample_bin_ids = np.digitize(x_test, q_bins)
            picps = np.array([picp(y_test[q_sample_bin_ids==bin], y_test_pred_lower_total[q_sample_bin_ids==bin],
                                   y_test_pred_upper_total[q_sample_bin_ids==bin]) for bin in range(num_bins)])
            unique_sample_bin_ids = np.digitize(x_uniques_sorted, q_bins)
            picp_replicated = [len(x_uniques_sorted[unique_sample_bin_ids == bin]) * [picps[bin]] for bin in range(num_bins)]
            picp_replicated = np.array([item for sublist in picp_replicated for item in sublist])
    else:
        picps = np.array([picp(y_test[sample_bin_ids == bin], y_test_pred_lower_total[sample_bin_ids == bin],
                                y_test_pred_upper_total[sample_bin_ids == bin]) for bin in range(num_unique)])
        picp_replicated = picps

    ax.plot(x_uniques_sorted, picp_replicated, label='PICP')
    ax.axhline(0.95, linestyle='--', label='95%')
    ax.set_ylabel('PICP')

    ax.legend(loc='best')

    if title is None:
        title = 'Test data overall PICP: {:.2f} MPIW: {:.2f}'.format(
                    picp(y_test, 
                        y_test_pred_lower_total, 
                        y_test_pred_upper_total),
                    mpiw(y_test_pred_lower_total, 
                        y_test_pred_upper_total))    

    if xlims is not None:
        ax.set_xlim(left=xlims[0], right=xlims[1])

    if ylims is not None:
        ax.set_ylim(bottom=ylims[0], top=ylims[1])
                
    ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)

    return ax


def plot_uncertainty_by_feature(x_test, y_test_pred_mean, y_test_pred_lower_total, y_test_pred_upper_total,
                                y_test_pred_lower_epistemic=None, y_test_pred_upper_epistemic=None,
                                ax=None, figsize=None, dpi=None, xlims=None, xscale="linear",
                                title=None, xlabel=None, ylabel=None):
    """
    Plot how prediction uncertainty varies across the entire range of a feature.

    Args:
        x_test: one dimensional ndarray.
            Feature column of the test dataset.
        y_test_pred_mean: One dimensional ndarray.
            Model prediction for the test dataset.
        y_test_pred_lower_total: One dimensional ndarray.
            Lower bound of the total uncertainty range.
        y_test_pred_upper_total: One dimensional ndarray.
            Upper bound of the total uncertainty range.
        y_test_pred_lower_epistemic: One dimensional ndarray.
            Lower bound of the epistemic uncertainty range.
        y_test_pred_upper_epistemic: One dimensional ndarray.
            Upper bound of the epistemic uncertainty range.
        ax: matplotlib.axes.Axes or None, optional (default=None). Target axes instance. If None, new figure and axes will be created.
        figsize: tuple of 2 elements or None, optional (default=None). Figure size.
        dpi : int or None, optional (default=None). Resolution of the figure.
        xlims : tuple of 2 elements or None, optional (default=None). Tuple passed to ``ax.xlim()``.
        xscale: Passed to ``ax.set_xscale()``.
        title : string or None, optional
            Axes title.
            If None, title is disabled.
        xlabel : string or None, optional
            X-axis title label.
            If None, title is disabled.
        ylabel : string or None, optional
            Y-axis title label.
            If None, title is disabled.

    Returns:
        matplotlib.axes.Axes: ax : The plot with model's uncertainty binned by a feature.

    """
    import matplotlib.pyplot as plt

    if ax is None:
        if figsize is not None:
            _check_not_tuple_of_2_elements(figsize, 'figsize')
        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    x_uniques_sorted = np.sort(np.unique(x_test))

    y_pred_var = ((y_test_pred_upper_total - y_test_pred_lower_total) / 4.0)**2
    agg_y_std = np.array([np.sqrt(np.mean(y_pred_var[x_test==x])) for x in x_uniques_sorted])
    agg_y_mean = np.array([np.mean(y_test_pred_mean[x_test==x]) for x in x_uniques_sorted])

    ax.plot(x_uniques_sorted, agg_y_mean, '-b', lw=2, label='mean prediction')
    ax.fill_between(x_uniques_sorted,
                     agg_y_mean - 2.0 * agg_y_std,
                     agg_y_mean + 2.0 * agg_y_std,
                     alpha=0.3, label='total uncertainty')

    if y_test_pred_lower_epistemic is not None:
        y_pred_var_epistemic = ((y_test_pred_upper_epistemic - y_test_pred_lower_epistemic) / 4.0)**2
        agg_y_std_epistemic = np.array([np.sqrt(np.mean(y_pred_var_epistemic[x_test==x])) for x in x_uniques_sorted])
        ax.fill_between(x_uniques_sorted,
                         agg_y_mean - 2.0 * agg_y_std_epistemic,
                         agg_y_mean + 2.0 * agg_y_std_epistemic,
                         alpha=0.3, label='model uncertainty')

    ax.legend(loc='best')
            
    if xlims is not None:
        ax.set_xlim(left=xlims[0], right=xlims[1]) 

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)

    return ax
