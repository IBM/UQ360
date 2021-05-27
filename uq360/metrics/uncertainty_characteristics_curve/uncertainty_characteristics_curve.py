from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps, trapz
from sklearn.isotonic import IsotonicRegression

DEFAULT_X_AXIS_NAME = 'excess'
DEFAULT_Y_AXIS_NAME = 'missrate'


class UncertaintyCharacteristicsCurve:
    """
    Class with main functions of the Uncertainty Characteristics Curve (UCC).

    """

    def __init__(self, normalize=True, precompute_bias_data=True):
        """
        :param normalize: set initial axes normalization flag (can be changed via set_coordinates())
        :param precompute_bias_data: if True, fit() will compute statistics necessary to generate bias-based
            UCCs (in addition to the scale-based ones). Skipping this precomputation may speed up the fit() call
            if bias-based UCC is not needed.

        """
        self.axes_name2idx = {"missrate": 1, "bandwidth": 2, "excess": 3, "deficit": 4}
        self.axes_idx2descr = {1: "Missrate", 2: "Bandwidth", 3: "Excess", 4: "Deficit"}
        self.x_axis_idx = None
        self.y_axis_idx = None
        self.norm_x_axis = False
        self.norm_y_axis = False
        self.std_unit = None
        self.normalize = normalize
        self.d = None
        self.gt = None
        self.lb = None
        self.ub = None
        self.precompute_bias_data = precompute_bias_data
        self.set_coordinates(x_axis_name=DEFAULT_X_AXIS_NAME, y_axis_name=DEFAULT_Y_AXIS_NAME, normalize=normalize)

    def set_coordinates(self, x_axis_name=None, y_axis_name=None, normalize=None):
        """
        Assigns user-specified type to the axes and normalization behavior (sticky).

        :param x_axis_name: None-> unchanged, or name from self.axes_name2idx 
        :param y_axis_name: ditto
        :param normalize: True/False will activate/deactivate norming for specified axes. Behavior for
                          Axes_name that are None will not be changed.
                          Value None will leave norm status unchanged.
                          Note, axis=='missrate' will never get normalized, even with normalize == True
        :return: none
        """
        normalize = self.normalize if normalize is None else normalize
        if x_axis_name is None and self.x_axis_idx is None:
            raise ValueError("ERROR(UCC): x-axis has not been defined.")
        if y_axis_name is None and self.y_axis_idx is None:
            raise ValueError("ERROR(UCC): y-axis has not been defined.")
        if x_axis_name is None and y_axis_name is None and normalize is not None:
            # just set normalization on/off for both axes and return
            self.norm_x_axis = False if x_axis_name == 'missrate' else normalize
            self.norm_y_axis = False if y_axis_name == 'missrate' else normalize
            return
        if x_axis_name is not None:
            self.x_axis_idx = self.axes_name2idx[x_axis_name]
            self.norm_x_axis = False if x_axis_name == 'missrate' else normalize
        if y_axis_name is not None:
            self.y_axis_idx = self.axes_name2idx[y_axis_name]
            self.norm_y_axis = False if y_axis_name == 'missrate' else normalize

    def set_std_unit(self, std_unit=None):
        """
        Sets the UCC's unit to be used when displaying normalized axes.

        :param std_unit: if None, the unit will be calculated as stddev of the ground truth data 
                        (ValueError raised if data has not been set at this point)
                        or set to the user-specified value.
        :return: 
        """
        if std_unit is None:  # set it to stddev of data
            if self.gt is None:
                raise ValueError("ERROR(UCC): No data specified - cannot set stddev unit.")
            self.std_unit = np.std(self.gt)

            if np.isclose(self.std_unit, 0.):
                print("WARN(UCC): data-based stddev is zero - resetting axes unit to 1.")
                self.std_unit = 1.
        else:
            self.std_unit = float(std_unit)

    def fit(self, X, gt):
        """
        Calculates internal arrays necessary for other methods (plotting, auc, cost minimization).
        Re-entrant.

        :param X: [numsamples, 3] numpy matrix, or list of numpy matrices.
                    Col 1: predicted values
                    Col 2: lower band (deviate) wrt predicted value  (always positive)
                    Col 3: upper band wrt predicted value  (always positive)
                    If list is provided, all methods will output corresponding metrics as lists as well!
        :param gt: Ground truth array (i.e.,the 'actual' values corresponding to predictions in X
        :return: self

        """
        if not isinstance(X, list):
            X = [X]
        newX = []
        for x in X:
            assert (isinstance(x, np.ndarray) and len(x.shape) == 2 and x.shape[1] == 3 and x.shape[0] == len(gt))
            newX.append(self._sanitize_input(x))
        self.d = [gt - x[:, 0] for x in newX]
        self.lb = [x[:, 1] for x in newX]
        self.ub = [x[:, 2] for x in newX]
        self.gt = gt
        self.set_std_unit()
        self.plotdata_for_scale = []
        self.plotdata_for_bias = []
        # precompute plotdata:
        for i in range(len(self.d)):
            self.plotdata_for_scale.append(self._calc_plotdata(self.d[i], self.lb[i], self.ub[i], vary_bias=False))
            if self.precompute_bias_data:
                self.plotdata_for_bias.append(self._calc_plotdata(self.d[i], self.lb[i], self.ub[i], vary_bias=True))

        return self

    def minimize_cost(self, x_axis_cost=.5, y_axis_cost=.5, augment_cost_by_normfactor=True,
                      search=('scale', 'bias')):
        """
        Find minima of a linear cost function for each component.
        Cost function C = x_axis_cost * x_axis_value + y_axis_cost * y_axis_value.
        A minimum can occur in the scale-based or bias-based UCC (this can be constrained by the 'search' arg).
        The function returns a 'recipe' how to achieve the corresponding minimum, for each component.

        :param x_axis_cost: weight of one unit on x_axis
        :param y_axis_cost: weight of one unit on y_axis
        :param augment_cost_by_normfactor: when False, the cost multipliers will apply as is. If True, they will be
            pre-normed by the corresponding axis norm (where applicable), to account for range differences between axes.
        :param search: list of types over which minimization is to be performed, valid elements are 'scale' and 'bias'.

        :return: list of dicts - one per component, or a single dict, if there is only one component. Dict keys are -
            'operation': can be 'bias' (additive) or 'scale' (multiplicative), 'modvalue': value to multiply by or to
            add to error bars to achieve the minimum, 'new_x'/'new_y': new coordinates (operating point) with that
            minimum, 'cost': new cost at minimum point, 'original_cost': original cost (original operating point).

        """
        if self.d is None:
            raise ValueError("ERROR(UCC): call fit() prior to using this method.")
        if augment_cost_by_normfactor:
            if self.norm_x_axis:
                x_axis_cost /= self.std_unit
            if self.norm_y_axis:
                y_axis_cost /= self.std_unit
            print("INFO(UCC): Pre-norming costs by corresp. std deviation: new x_axis_cost = %.4f, y_axis_cost = %.4f" %
                  (x_axis_cost, y_axis_cost))
        if isinstance(search, tuple):
            search = list(search)
        if not isinstance(search, list):
            search = [search]

        min_costs = []
        for d in range(len(self.d)):
            # original OP cost
            m, b, e, df = self._calc_missrate_bandwidth_excess_deficit(self.d[d], self.lb[d], self.ub[d])
            original_cost = x_axis_cost * [0., m, b, e, df][self.x_axis_idx] + y_axis_cost * [0., m, b, e, df][
                self.y_axis_idx]

            plotdata = self.plotdata_for_scale[d]
            cost_scale, minidx_scale = self._find_min_cost_in_component(plotdata, self.x_axis_idx, self.y_axis_idx,
                                                                        x_axis_cost, y_axis_cost)
            mcf_scale_multiplier = plotdata[minidx_scale][0]
            mcf_scale_x = plotdata[minidx_scale][self.x_axis_idx]
            mcf_scale_y = plotdata[minidx_scale][self.y_axis_idx]

            if 'bias' in search:
                if not self.precompute_bias_data:
                    raise ValueError(
                        "ERROR(UCC): Cannot perform minimization - instantiated without bias data computation")
                plotdata = self.plotdata_for_bias[d]
                cost_bias, minidx_bias = self._find_min_cost_in_component(plotdata, self.x_axis_idx, self.y_axis_idx,
                                                                          x_axis_cost, y_axis_cost)
                mcf_bias_add = plotdata[minidx_bias][0]
                mcf_bias_x = plotdata[minidx_bias][self.x_axis_idx]
                mcf_bias_y = plotdata[minidx_bias][self.y_axis_idx]

            if 'bias' in search and 'scale' in search:
                if cost_bias < cost_scale:
                    min_costs.append({'operation': 'bias', 'cost': cost_bias, 'modvalue': mcf_bias_add,
                                      'new_x': mcf_bias_x, 'new_y': mcf_bias_y, 'original_cost': original_cost})
                else:
                    min_costs.append({'operation': 'scale', 'cost': cost_scale, 'modvalue': mcf_scale_multiplier,
                                      'new_x': mcf_scale_x, 'new_y': mcf_scale_y, 'original_cost': original_cost})
            elif 'scale' in search:
                min_costs.append({'operation': 'scale', 'cost': cost_scale, 'modvalue': mcf_scale_multiplier,
                                  'new_x': mcf_scale_x, 'new_y': mcf_scale_y, 'original_cost': original_cost})
            elif 'bias' in search:
                min_costs.append({'operation': 'bias', 'cost': cost_bias, 'modvalue': mcf_bias_add,
                                  'new_x': mcf_bias_x, 'new_y': mcf_bias_y, 'original_cost': original_cost})
            else:
                raise ValueError("(ERROR): Unknown search element (%s) requested." % ",".join(search))

        if len(min_costs) < 2:
            return min_costs[0]
        else:
            return min_costs

    def get_specific_operating_point(self, req_x_axis_value=None, req_y_axis_value=None,
                                     req_critical_value=None, vary_bias=False):
        """
        Finds corresponding operating point on the current UCC, given a point on either x or y axis. Returns
        a list of recipes how to achieve the point (x,y), for each component. If there is only one component,
        returns a single recipe dict.

        :param req_x_axis_value: requested x value on UCC (normalization status is taken from current display)
        :param req_y_axis_value: requested y value on UCC (normalization status is taken from current display)
        :param vary_bias: set to True when referring to bias-induced UCC (scale UCC default)
        :return: list of dicts (recipes), or a single dict
        """
        if self.d is None:
            raise ValueError("ERROR(UCC): call fit() prior to using this method.")
        if np.sum([req_x_axis_value is not None, req_y_axis_value is not None, req_critical_value is not None]) != 1:
            raise ValueError("ERROR(UCC): exactly one axis value must be requested at a time.")
        if vary_bias and not self.precompute_bias_data:
            raise ValueError("ERROR(UCC): Cannot vary bias - instantiated without bias data computation")
        xnorm = self.std_unit if self.norm_x_axis else 1.
        ynorm = self.std_unit if self.norm_y_axis else 1.
        recipe = []
        for dc in range(len(self.d)):
            plotdata = self.plotdata_for_bias[dc] if vary_bias else self.plotdata_for_scale[dc]
            if req_x_axis_value is not None:
                tgtidx = self.x_axis_idx
                req_value = req_x_axis_value * xnorm
            elif req_y_axis_value is not None:
                tgtidx = self.y_axis_idx
                req_value = req_y_axis_value * ynorm
            elif req_critical_value is not None:
                req_value = req_critical_value
                tgtidx = 0  # first element in plotdata is always the critical value (scale of bias)
            else:
                raise RuntimeError("Unhandled case")
            closestidx = np.argmin(np.asarray([np.abs(p[tgtidx] - req_value) for p in plotdata]))
            recipe.append({'operation': ('bias' if vary_bias else 'scale'),
                           'modvalue': plotdata[closestidx][0],
                           'new_x': plotdata[closestidx][self.x_axis_idx] / xnorm,
                           'new_y': plotdata[closestidx][self.y_axis_idx] / ynorm})
        if len(recipe) < 2:
            return recipe[0]
        else:
            return recipe


    def _find_min_cost_in_component(self, plotdata, idx1, idx2, cost1, cost2):
        """
        Find s minimum cost function value and corresp. position index in plotdata

        :param plotdata: liste of tuples
        :param idx1: idx of  x-axis item within the tuple
        :param idx2: idx of y-axis item within the tuple
        :param cost1: cost factor for x-axis unit
        :param cost2: cost factor for y-axis unit
        :return: min cost value, index within plotdata where minimum occurs
        """
        raw = [cost1 * i[idx1] + cost2 * i[idx2] for i in plotdata]
        minidx = np.argmin(raw)
        return raw[minidx], minidx

    def _sanitize_input(self, x):
        """
        Replaces problematic values in input data (e.g, zero error bars)

        :param x: single matrix of input data [n, 3]
        :return: sanitized version of x
        """
        if np.isclose(np.sum(x[:, 1]), 0.):
            raise ValueError("ERROR(UCC): Provided lower bands are all zero.")
        if np.isclose(np.sum(x[:, 2]), 0.):
            raise ValueError("ERROR(UCC): Provided upper bands are all zero.")
        for i in [1, 2]:
            if any(np.isclose(x[:, i], 0.)):
                print("WARN(UCC): some band values are 0. - REPLACING with positive minimum")
                m = np.min(x[x[:, i] > 0, i])
                x = np.where(np.isclose(x, 0.), m, x)
        return x

    def _calc_avg_excess(self, d, lb, ub):
        """
        Excess is amount an error bar overshoots actual

        :param d: pred-actual array
        :param lb: lower band
        :param ub: upper band
        :return: average excess over array
        """
        excess = np.zeros(d.shape)
        posidx = np.where(d >= 0)[0]
        excess[posidx] = np.where(ub[posidx] - d[posidx] < 0., 0., ub[posidx] - d[posidx])
        negidx = np.where(d < 0)[0]
        excess[negidx] = np.where(lb[negidx] + d[negidx] < 0., 0., lb[negidx] + d[negidx])
        return np.mean(excess)

    def _calc_avg_deficit(self, d, lb, ub):
        """
        Deficit is error bar insufficiency: bar falls short of actual

        :param d: pred-actual array
        :param lb: lower band
        :param ub: upper band
        :return: average deficit over array
        """
        deficit = np.zeros(d.shape)
        posidx = np.where(d >= 0)[0]
        deficit[posidx] = np.where(- ub[posidx] + d[posidx] < 0., 0., - ub[posidx] + d[posidx])
        negidx = np.where(d < 0)[0]
        deficit[negidx] = np.where(- lb[negidx] - d[negidx] < 0., 0., - lb[negidx] - d[negidx])
        return np.mean(deficit)

    def _calc_missrate_bandwidth_excess_deficit(self, d, lb, ub, scale=1.0, bias=0.0):
        """
        Calculates recall at a given scale/bias, average bandwidth and average excess

        :param d: delta
        :param lb: lower band
        :param ub: upper band
        :param scale: scale * (x + bias)
        :param bias:
        :return: miss rate, average bandwidth, avg excess, avg deficit
        """
        abslband = scale * np.where((lb + bias) < 0., 0., lb + bias)
        absuband = scale * np.where((ub + bias) < 0., 0., ub + bias)
        recall = np.sum((d >= - abslband) & (d <= absuband)) / len(d)
        avgbandwidth = np.mean([absuband, abslband])
        avgexcess = self._calc_avg_excess(d, abslband, absuband)
        avgdeficit = self._calc_avg_deficit(d, abslband, absuband)
        return 1 - recall, avgbandwidth, avgexcess, avgdeficit

    def _calc_plotdata(self, d, lb, ub, vary_bias=False):
        """
        Generates data necessary for various UCC metrics.

        :param d: delta (predicted - actual) vector
        :param ub: upper uncertainty bandwidth (above predicted)
        :param lb: lower uncertainty bandwidth (below predicted) - all positive (bandwidth)
        :param vary_bias: True will switch to additive bias instead of scale
        :return: list. Elements are tuples (varyvalue, missrate, bandwidth, excess, deficit)
        """

        # step 1: collect critical scale or bias values
        critval = []
        for i in range(len(d)):
            if not vary_bias:
                if d[i] >= 0:
                    critval.append(d[i] / ub[i])
                else:
                    critval.append(-d[i] / lb[i])
            else:
                if d[i] >= 0:
                    critval.append(d[i] - ub[i])
                else:
                    critval.append(-lb[i] - d[i])
        critval = sorted(critval)
        plotdata = []
        for i in range(len(critval)):
            if not vary_bias:
                missrate, bandwidth, excess, deficit = self._calc_missrate_bandwidth_excess_deficit(d, lb, ub,
                                                                                                    scale=critval[i])
            else:
                missrate, bandwidth, excess, deficit = self._calc_missrate_bandwidth_excess_deficit(d, lb, ub,
                                                                                                    bias=critval[i])
            plotdata.append((critval[i], missrate, bandwidth, excess, deficit))

        return plotdata

    def get_AUUCC(self, vary_bias=False, aucfct="trapz", partial_x=None, partial_y=None):
        """
        returns approximate area under the curve on current coordinates, for each component.

        :param vary_bias: False == varies scale, True == varies bias
        :param aucfct: specifies AUC integrator (can be "trapz", "simps")
        :param partial_x: tuple (x_min, x_max) defining interval on x to calc a a partial AUC.
                        The interval bounds refer to axes as visualized (ie. potentially normed)
        :param partial_y: tuple (y_min, y_max) defining interval on y to calc a a partial AUC. partial_x must be None.
        :return: list of floats with AUUCCs for each input component, or a single float, if there is only 1 component.
        """
        if self.d is None:
            raise ValueError("ERROR(UCC): call fit() prior to using this method.")
        if vary_bias and not self.precompute_bias_data:
            raise ValueError("ERROR(UCC): Cannot vary bias - instantiated without bias data computation")
        if partial_x is not None and partial_y is not None:
            raise ValueError("ERROR(UCC): partial_x and partial_y can not be specified at the same time.")
        assert(partial_x is None or (isinstance(partial_x, tuple) and len(partial_x)==2))
        assert(partial_y is None or (isinstance(partial_y, tuple) and len(partial_y)==2))

        # find starting point (where the x axis value starts to actually change)
        rv = []
        # do this for individual streams 
        xind = self.x_axis_idx
        aucfct = simps if aucfct == "simps" else trapz
        for s in range(len(self.d)):
            plotdata = self.plotdata_for_bias[s] if vary_bias else self.plotdata_for_scale[s]
            prev = plotdata[0][xind]
            t = 1
            cval = plotdata[t][xind]
            while cval == prev and t < len(plotdata) - 1:
                t += 1
                prev = cval
                cval = plotdata[t][xind]
            startt = t - 1  # from here, it's a valid function
            endtt = len(plotdata)

            if startt >= endtt - 2:
                rvs = 0.  # no area
            else:
                xnorm = self.std_unit if self.norm_x_axis else 1.
                ynorm = self.std_unit if self.norm_y_axis else 1.
                y=[(plotdata[i][self.y_axis_idx]) / ynorm for i in range(startt, endtt)]
                x=[(plotdata[i][self.x_axis_idx]) / xnorm for i in range(startt, endtt)]
                if partial_x is not None:
                    from_i = self._find_closest_index(partial_x[0], x)
                    to_i = self._find_closest_index(partial_x[1], x) + 1
                elif partial_y is not None:
                    from_i = self._find_closest_index(partial_y[0], y)
                    to_i = self._find_closest_index(partial_y[1], y)
                    if from_i > to_i:   # y is in reverse order
                        from_i, to_i = to_i, from_i
                    to_i += 1 # as upper bound in array indexing
                else:
                    from_i = 0
                    to_i = len(x)
                to_i = min(to_i, len(x))
                if to_i < from_i:
                    raise ValueError("ERROR(UCC): Failed to find an appropriate partial-AUC interval in the data.")
                if to_i - from_i < 2:
                    raise RuntimeError("ERROR(UCC): There are too few samples (1) in the partial-AUC interval specified")
                rvs = aucfct(x=x[from_i:to_i], y=y[from_i:to_i])
            rv.append(rvs)
        if len(rv) < 2:
            return rv[0]
        else:
            return rv

    @ staticmethod
    def _find_closest_index(value, array):
        """
        Returns an index of the 'array' element closest in value to 'value'

        :param value:
        :param array:
        :return:
        """
        return np.argmin(np.abs(np.asarray(array)-value))

    def _get_single_OP(self, d, lb, ub, scale=1., bias=0.):
        """
        Returns Operating Point for original input data, on coordinates currently set up, given a scale/bias.

        :param scale:
        :param bias: 
        :return: single tuple (x point, y point, unit of x, unit of y)
        """
        xnorm = self.std_unit if self.norm_x_axis else 1.
        ynorm = self.std_unit if self.norm_y_axis else 1.
        auxop = self._calc_missrate_bandwidth_excess_deficit(d, lb, ub, scale=scale, bias=bias)
        op = [0.] + [i for i in auxop]  # mimic plotdata (first element ignored here)
        return (op[self.x_axis_idx] / xnorm, op[self.y_axis_idx] / ynorm, xnorm, ynorm)

    def get_OP(self, scale=1., bias=0.):
        """
        Returns all Operating Points for original input data, on coordinates currently set up, given a scale/bias.

        :param scale:
        :param bias:
        :return: list of tuples (x point, y point, unit of x, unit of y) or a single tuple if there is only
                1 component.
        """
        if self.d is None:
            raise ValueError("ERROR(UCC): call fit() prior to using this method.")
        op = []
        for dc in range(len(self.d)):
            op.append(self._get_single_OP(self.d[dc], self.lb[dc], self.ub[dc], scale=scale, bias=bias))
        if len(op) < 2:
            return op[0]
        else:
            return op

    def plot_UCC(self, titlestr='', syslabel='model', outfn=None, vary_bias=False, markers=None,
                 xlim=None, ylim=None, **kwargs):
        """ Will plot/display the UCC based on current data and coordinates. Multiple curves will be shown
        if there are multiple data components (via fit())

        :param titlestr: Plot title string
        :param syslabel: list is label strings to appear in the plot legend. Can be single, if one component.
        :param outfn: base name of an image file to be created (will append .png before creating)
        :param vary_bias: True will switch to varying additive bias (default is multiplicative scale)
        :param markers: None or a list of marker styles to be used for each curve.
            List must be same or longer than number of components.
            Markers can be one among these ['o', 's', 'v', 'D', '+'].
        :param xlim: tuples or lists of specifying the range for the x axis, or None (auto)
        :param ylim: tuples or lists of specifying the range for the y axis, or None (auto)
        :param `**kwargs`: Additional arguments passed to the main plot call.

        :return: list of areas under the curve (or single area, if one data component)
                 list of operating points (or single op): format of an op is tuple (xaxis value, yaxis value, xunit, yunit)
        """
        if self.d is None:
            raise ValueError("ERROR(UCC): call fit() prior to using this method.")
        if vary_bias and not self.precompute_bias_data:
            raise ValueError("ERROR(UCC): Cannot vary bias - instantiated without bias data computation")
        if not isinstance(syslabel, list):
            syslabel = [syslabel]
        assert (len(syslabel) == len(self.d))
        assert (markers is None or (isinstance(markers, list) and len(markers) >= len(self.d)))
        # main plot of (possibly multiple) datasets
        plt.figure()
        xnorm = self.std_unit if self.norm_x_axis else 1.
        ynorm = self.std_unit if self.norm_y_axis else 1.
        op_info = []
        auucc = self.get_AUUCC(vary_bias=vary_bias)
        auucc = [auucc] if not isinstance(auucc, list) else auucc
        for s in range(len(self.d)):
            # original operating point
            x_op, y_op, x_unit, y_unit = self._get_single_OP(self.d[s], self.lb[s], self.ub[s])
            op_info.append((x_op, y_op, x_unit, y_unit))
            # display chart
            plotdata = self.plotdata_for_scale[s] if not vary_bias else self.plotdata_for_bias[s]
            axisX_data = [i[self.x_axis_idx] / xnorm for i in plotdata]
            axisY_data = [i[self.y_axis_idx] / ynorm for i in plotdata]
            marker = None
            if markers is not None: marker = markers[s]
            p = plt.plot(axisX_data, axisY_data, label=syslabel[s] + (" (AUC=%.3f)" % auucc[s]), marker=marker, **kwargs)
            if s + 1 == len(self.d):
                oplab = 'OP'
            else:
                oplab = None
            plt.plot(x_op, y_op, marker='o', color=p[0].get_color(), label=oplab, markerfacecolor='w',
                     markeredgewidth=1.5, markeredgecolor=p[0].get_color())
        axisX_label = self.axes_idx2descr[self.x_axis_idx]
        axisY_label = self.axes_idx2descr[self.y_axis_idx]
        axisX_units = "(raw)" if np.isclose(xnorm, 1.0) else "[in std deviations]"
        axisY_units = "(raw)" if np.isclose(ynorm, 1.0) else "[in std deviations]"
        axisX_label += ' ' + axisX_units
        axisY_label += ' ' + axisY_units
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        plt.xlabel(axisX_label)
        plt.ylabel(axisY_label)
        plt.legend()
        plt.title(titlestr)
        plt.grid()
        if outfn is None:
            plt.show()
        else:
            plt.savefig(outfn)
        if len(auucc) < 2:
            auucc = auucc[0]
            op_info = op_info[0]
        return auucc, op_info
