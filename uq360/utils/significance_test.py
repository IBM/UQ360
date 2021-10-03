
import math

import numpy as np
from scipy.stats import norm



class SignificanceTester:
    """
    Class for non-parametric significance testing. It has two main functions:

    hypothesis_test: Assumes two vectors of paired observations (ie each point in measurement_1 is paired with the point
    with the same index in measurement_2, for example measurements of the length of the same set of objects using two
    different rulers). Performs a permutation test to determine if one set of measurements is statistically different
    (higher, lower, or either one, depending on the 'tailed' argument) from the other, and computes a p-value.

    confidence_interval: Uses bootstrapping to compute a confidence interval (controlled by 'alpha' argument, default is
    95% confidence interval) for a list of (1-dimensional) measurements.
    """
    def __init__(self, metric):
        self.metric = self.get_metric_function(metric)

    def hypothesis_test(self, measurement_1, measurement_2, metric_payload={}, n_iter=1000, tailed='two', verbose=False):
        result = self._permutation_test(measurement_1, measurement_2, metric_payload=metric_payload, 
                                        n_iter=n_iter, tailed=tailed, verbose=verbose)
        return result

    def confidence_interval(self, measurement, metric_payload={}, metric_kwargs={}, n_iter=10000, alpha=0.05, verbose=False):
        result = self._bootstrap_test(measurement, metric_payload=metric_payload, metric_kwargs=metric_kwargs, 
                                        n_iter=n_iter, alpha=alpha, verbose=verbose)
        return result

    # This computes the BCa version of a bootstrap confidence interval
    def _bootstrap_test(self, measurement, metric_payload={}, metric_kwargs={}, n_iter=10000, alpha=0.05, verbose=False):
        if verbose:
            print("Computing bootstrap {} confidence intervals with {} iterations".format(alpha, n_iter))
        n_samples = len(measurement)

        theta_hat = self.resample_compute(measurement, np.arange(n_samples), metric_payload, metric_kwargs)
        
        bootstrap_measurements = []
        for n in range(n_iter):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples)
            theta_b = self.resample_compute(measurement, bootstrap_indices, metric_payload, metric_kwargs)
            bootstrap_measurements.append(theta_b)
            if verbose:
                if n % 1000 == 0:
                    print("Iteration {}, value = {}".format(n, theta_b))

        sorted_bootstraps = sorted(bootstrap_measurements)
        # All values are the same
        difference = (sorted_bootstraps[-1]-sorted_bootstraps[0]) / (0.5*(sorted_bootstraps[-1] + sorted_bootstraps[0]))
        if difference < 0.001:
            print()
            print()
            print("BOOTSTRAP CONFIDENCE INTERVALS CANNOT BE COMPUTED WITH DIFFERENCE OF LESS THAN 0.01%")
            return sorted_bootstraps[0], sorted_bootstraps[0], sorted_bootstraps[0]
        elif not sorted_bootstraps[0] < sorted_bootstraps[-1]:
            return sorted_bootstraps[0], sorted_bootstraps[0], sorted_bootstraps[0]

        # Get uncorrected Z-stats
        Z_low = norm.ppf(alpha/2.0)
        Z_high = norm.ppf(1.0 - alpha/2.0)

        # Get the bias correction factor. 
        bias_correction = sum([1 if x < theta_hat else 0 for x in sorted_bootstraps]) / float(n_iter)
        bias_correction = norm.ppf(bias_correction)
        # Use jackknife estimation of acceleration factor
        jackknife_measurements = []
        for i in range(n_samples):
            jackknife_indices = np.concatenate((np.arange(n_samples)[:i], np.arange(n_samples)[i+1:]))
            theta_j = self.resample_compute(measurement, jackknife_indices, metric_payload, metric_kwargs)
            jackknife_measurements.append(theta_j)
        
        C1 = sum([(theta_hat - x)**3 for x in jackknife_measurements])
        C2 = sum([(theta_hat - x)**2 for x in jackknife_measurements])
        acceleration = C1 / (6 * (C2**1.5))

        # Combine the pieces
        Z_low_corrected = bias_correction + (bias_correction + Z_low) / (1.0 - acceleration * (bias_correction + Z_low))
        Z_high_corrected = bias_correction + (bias_correction + Z_high) / (1.0 - acceleration * (bias_correction + Z_high))
        alpha_low = norm.cdf(Z_low_corrected)
        alpha_high = norm.cdf(Z_high_corrected)

        ind_low = int(math.ceil(n_iter*alpha_low))
        ind_high = int(math.floor(n_iter*alpha_high))
        theta_low = sorted_bootstraps[ind_low]
        theta_high = sorted_bootstraps[ind_high]
        return theta_hat, theta_low, theta_high

    def resample_compute(self, measurement, indices, metric_payload, metric_kwargs):
        new_payload = {}
        for key, val in metric_payload.items():
            new_payload[key] = np.array(val)[indices]
        for key, val in metric_kwargs.items():
            new_payload[key] = val
        theta_resampled = self.metric(np.array(measurement)[indices], **new_payload)
        return theta_resampled

    """
    measurement_1 and measurement_2 are 1-D arrays of equal length holding the (paired) values of some measurement 
    metric_function is a function which computes the desired statistic, and metric_payload holds kwargs for this function. 
    permutation_test will perform n_iter permutations, and output a p-value for either a two-tailed test, a one-tailed test, or both
    """
    def _permutation_test(self, measurement_1, measurement_2, metric_payload={}, n_iter=1000, tailed='two', verbose=False):
        assert len(measurement_1) == len(measurement_2)
        print("Starting {}-tailed permutation test with {} iterations".format(tailed, n_iter))
        base_1 = self.metric(measurement_1, **metric_payload)
        base_2 = self.metric(measurement_2, **metric_payload)
        
        values_1 = []
        values_2 = []
        for n in range(n_iter):
            binom = np.random.binomial(n=1, p=0.5, size=len(measurement_1))
            permuted_values_1 = [u1 if b else u2 for u1, u2, b in zip(measurement_1, measurement_2, binom)]
            permuted_values_2 = [u2 if b else u1 for u1, u2, b in zip(measurement_1, measurement_2, binom)]
            m1 = self.metric(permuted_values_1, **metric_payload)
            m2 = self.metric(permuted_values_2, **metric_payload)
            if verbose:
                if n % 1000 == 0:
                    print("Iteration {}, values = {}, {}".format(n, m1, m2))
            values_1.append(m1)
            values_2.append(m2)
        
        if tailed == 'two':
            diffs = [abs(v1 - v2) for v1, v2 in zip(values_1, values_2)]
            pval = (sum([d > abs(base_1 - base_2) for d in diffs]) + 1) / (n_iter + 1)
            return pval, base_1, base_2
        elif tailed == 'one':
            diffs = [v1 - v2 for v1, v2 in zip(values_1, values_2)]
            pval = (sum([d > base_1 - base_2 for d in diffs]) + 1) / (n_iter + 1)
            return pval, base_1, base_2
        elif tailed == 'both':
            diffs1 = [abs(v1 - v2) for v1, v2 in zip(values_1, values_2)]
            pval1 = (sum([d > abs(base_1 - base_2) for d in diffs1]) + 1) / (n_iter + 1)

            diffs2 = [v1 - v2 for v1, v2 in zip(values_1, values_2)]
            pval2 = (sum([d > base_1 - base_2 for d in diffs2]) + 1) / (n_iter + 1)
            return pval1, pval2, base_1, base_2
        else:
            raise Exception("argument 'tailed' must be either 'one', 'two', or 'both'")

    def get_metric_function(self, metric):
        assert metric in ['LCMR', 'cost', 'auucc', 'average']
        if metric == 'cost':
            return self.get_cost
        elif metric == 'average':
            return self.get_average

    def get_cost(self, predictions, deltas=[], ratio=0.5, exp=1):
        assert len(predictions) == len(deltas)
        cost = 0.0
        overshoot_weight = (1.0 - ratio)
        undershoot_weight = ratio
        for d, p in zip(deltas, predictions):
            assert d >=0
            assert p >=0
            if d <= p:
                cost += overshoot_weight * (p - d)**exp
            else:
                cost += undershoot_weight * (d - p)**exp
        cost /= float(len(deltas))
        return cost

    def get_average(self, predictions):
        return np.mean(np.array(predictions))
