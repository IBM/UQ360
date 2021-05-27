import unittest
from unittest import TestCase
import numpy as np


class TestUncertaintyCharacteristicsCurve(TestCase):
    def _generate_mock_data(self):
        errors = [1.3642327303743222, 1.90992286562998, 1.3376553033984742, 1.1360514041212681, 1.0059398687783236,
                  0.6562763187757668, 0.583628028840792, 0.6876683476085894, 1.0506101454179664, 0.795072119831687,
                  1.3275374841578582, 2.4458894373634283, 2.909916525881682, 2.837773991026335, 2.550841867998461]
        pred = [96.98915353045395, 96.23007611746924, 97.59180156001409, 98.50349438071208, 97.58943114819733,
                97.90442496968824, 97.76155157998329, 96.83695266121595, 96.33141125146022, 96.67439930495053,
                95.17170073977303, 92.64549009869268, 94.29582588015835, 96.04923654039105, 97.50696433632777]
        actual = [99.02260232131948, 98.61111111111111, 98.03439803439804, 98.05194805194805, 97.47126436781609,
                  97.27272727272727, 96.75675675675676, 96.328125, 95.68627450980392, 95.0, 94.48051948051948,
                  94.78260869565217, 93.75433726578764, 93.59375, 76.30066780555987]
        X = np.zeros([len(errors), 3])
        X[:, 0] = pred
        X[:, 1] = errors
        X[:, 2] = errors
        return X, actual

    def test_set_coordinates(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        o = ucc()
        o.set_coordinates(x_axis_name='missrate', y_axis_name='bandwidth', normalize=True)
        assert(o.norm_x_axis==False and o.norm_y_axis==True)
        assert(o.x_axis_idx==o.axes_name2idx['missrate'])
        assert(o.y_axis_idx==o.axes_name2idx['bandwidth'])
        o.set_coordinates(x_axis_name='bandwidth', y_axis_name='missrate', normalize=True)
        assert(o.norm_x_axis==True and o.norm_y_axis==False)
        assert(o.x_axis_idx==o.axes_name2idx['bandwidth'])
        assert(o.y_axis_idx==o.axes_name2idx['missrate'])
        o.set_coordinates(x_axis_name='excess', y_axis_name='deficit', normalize=False)
        assert(o.norm_x_axis==False and o.norm_y_axis==False)
        assert(o.x_axis_idx==o.axes_name2idx['excess'])
        assert(o.y_axis_idx==o.axes_name2idx['deficit'])

    def test_set_std_unit(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        o = ucc()
        self.assertRaises(ValueError, o.set_std_unit)
        X, gt = self._generate_mock_data()
        o.fit(X, gt)
        o.set_std_unit()
        assert (np.isclose(np.std(gt), o.std_unit))

    def test_fit(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        X, gt = self._generate_mock_data()
        o = ucc()
        o.fit(X, gt)
        assert (len(o.d) == 1)
        assert (all(np.isclose(o.d[0], gt - X[:, 0])))
        assert (all(np.isclose(o.lb[0], X[:, 1])))
        o.fit([X, X], gt)
        assert (len(o.d) == 2)

    def test__sanitize_input(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        from copy import deepcopy
        X, gt = self._generate_mock_data()
        o = ucc()
        x = deepcopy(X)
        x[:, 1:] = 0.
        self.assertRaises(ValueError, o._sanitize_input, x)
        x = deepcopy(X)
        x[0:2, 1] = 0.
        x[2:4, 2] = 0.
        x = o._sanitize_input(x)
        assert all(x[0:2, 1] != 0)
        assert all(x[2:4, 2] != 0)

    def test__calc_missrate_bandwidth_excess_deficit(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        X, gt = self._generate_mock_data()
        o = ucc()
        d = X[:, 0] - gt
        lb = X[:, 1]
        ub = X[:, 2]
        m, b, e, df = o._calc_missrate_bandwidth_excess_deficit(d, lb, ub)
        assert (np.isclose(m, 0.333333) and np.isclose(b, 1.506601) and np.isclose(e, 0.451471) and np.isclose(df,
                                                                                                               1.406418))

    def test__calc_plotdata(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        X, gt = self._generate_mock_data()
        o = ucc()
        d = X[:, 0] - gt
        lb = X[:, 1]
        ub = X[:, 2]
        pd = o._calc_plotdata(d, lb, ub)
        assert (len(pd) == 15)
        assert (all(np.isclose(pd[-1], (8.313450079681967, 0.0, 12.525053001149491, 10.06350492529685, 0.0))))
        pd = o._calc_plotdata(d, lb, ub, vary_bias=True)
        assert (len(pd) == 15)
        assert (all(np.isclose(pd[-1], (18.65545466276944, 0.0, 20.162055758716438, 17.70050768286379, 0.0))))

    def test_get_AUUCC_OP(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        X, gt = self._generate_mock_data()
        o = ucc()
        o.fit(X, gt)
        assert (np.isclose(o.get_AUUCC(), 0.14510967778953665))
        assert (np.isclose(o.get_AUUCC(vary_bias=True), 0.1818591455455332))
        d = X[:, 0] - gt
        lb = X[:, 1]
        ub = X[:, 2]
        op = o._get_single_OP(d, lb, ub)
        assert (all(np.isclose(op, (0.08553963270429753, 0.33333333333333337, 5.2779216043654325, 1.0))))
        op = o._get_single_OP(d, lb, ub, 2.0, 0.5)
        assert (all(np.isclose(op, (0.4847795200753158, 0.06666666666666665, 5.2779216043654325, 1.0))))
        o.set_coordinates('excess', 'deficit', normalize=True)
        op = o._get_single_OP(d, lb, ub)
        assert (all(np.isclose(op, (0.08553963270429753, 0.26647202456017466, 5.2779216043654325, 5.2779216043654325))))
        o.set_coordinates('excess', 'deficit', normalize=False)
        op = o._get_single_OP(d, lb, ub)
        assert (all(np.isclose(op, (0.45147147547949584, 1.406418455385142, 1.0, 1.0))))


    def test_get_partial_AUUCC(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        X, gt = self._generate_mock_data()
        o = ucc()
        o.set_coordinates(x_axis_name='bandwidth', y_axis_name='missrate', normalize=True)
        o.fit(X, gt)
        assert (np.isclose(o.get_AUUCC(partial_y=(0.,0.1)), .07003588145613951))
        assert (np.isclose(o.get_AUUCC(partial_x=(0.,.5)), 0.21031874267804815))
        assert (np.isclose(o.get_AUUCC(), o.get_AUUCC(partial_y=(0.,1.))))
        assert (np.isclose(o.get_AUUCC(), o.get_AUUCC(partial_x=(0.,1000000.))))
        self.assertRaises(ValueError, o.get_AUUCC, partial_x=(0., 1.), partial_y=(0., 1.))

        # assert (np.isclose(o.get_AUUCC(vary_bias=True), 0.1818591455455332))


    def test_minimize_cost_and_get_recipe(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        from copy import deepcopy
        X, gt = self._generate_mock_data()
        o = ucc()
        o.fit(X, gt)
        C = o.minimize_cost(x_axis_cost=1.0, y_axis_cost=10., augment_cost_by_normfactor=False)
        assert(C['operation']=='bias' and np.isclose(C['cost'], 1.7761220370565665) and np.isclose(C['modvalue'], 0.8793271851188393))
        Cn = o.minimize_cost(x_axis_cost=1.0, y_axis_cost=10., augment_cost_by_normfactor=True)
        # Cn should have different cost value but same optimum coordinates
        assert(Cn['modvalue']==C['modvalue'] and Cn['new_x']==C['new_x'] and Cn['new_y']==C['new_y']
               and Cn['operation']==C['operation'])
        X2 = deepcopy(X)
        X2[:,1:] = X[:,1:] + C['modvalue']
        o2 = ucc()
        o2.fit(X2, gt)
        C2 = o2.minimize_cost(x_axis_cost=1.0, y_axis_cost=10., augment_cost_by_normfactor=False)
        assert(np.isclose(C2['cost'], C2['original_cost']))
        r = o2.get_specific_operating_point(req_x_axis_value=C2['new_x'], vary_bias=True)
        assert(np.isclose(r['new_y'], C2['new_y']))
        r2 = o2.get_specific_operating_point(req_x_axis_value=C2['new_x'], vary_bias=False)
        assert(r2['new_y'] >= r['new_y'])  # scale ucc happens to have higher cost
        r = o2.get_specific_operating_point(req_y_axis_value=C2['new_y'], vary_bias=True)
        assert(r['new_x']<=C2['new_x'])   # if multiple x's for a y, the lowest is returned
        r2 = o2.get_specific_operating_point(req_y_axis_value=C2['new_y'], vary_bias=False)
        assert(np.isclose(r2['new_x'], r['new_x']))  # x points should be the same
        assert(np.isclose(r['modvalue'], 0.,) and np.isclose(r2['modvalue'], 1.))
        op = o2.get_OP()
        assert(np.isclose(op[0] * op[2] * 1. + op[1] * op[3] * 10., C2['original_cost']))
        # test normalization
        o2.set_coordinates(normalize=False)
        r3 = o2.get_specific_operating_point(req_y_axis_value=C2['new_y'])
        assert(np.isclose(r3['new_x'] / o2.std_unit, r2['new_x']))
        # test enforcing optimization
        C3 = o.minimize_cost(x_axis_cost=1.0, y_axis_cost=10., augment_cost_by_normfactor=False, search=['scale'])
        C4 = o.minimize_cost(x_axis_cost=1.0, y_axis_cost=10., augment_cost_by_normfactor=False, search=['bias'])
        assert(C3['operation']=='scale')
        assert(C4['operation']=='bias')
        assert(C3['cost']>C4['cost'])  # we know in this example scale has higher cost
        assert(np.isclose(C4['cost'], C['cost']))
        # no bias precomputed, test stuff just for scale
        o = ucc(precompute_bias_data=False)
        o.fit(X, gt)
        o.minimize_cost(search=['scale'])
        self.assertRaises(ValueError, o.minimize_cost, search=['bias'])


    def test_multiple_components(self):
        from uq360.metrics.uncertainty_characteristics_curve.uncertainty_characteristics_curve import UncertaintyCharacteristicsCurve as ucc
        from copy import deepcopy
        X, gt = self._generate_mock_data()
        X2 = deepcopy(X)
        X3 = deepcopy(X)
        X4 = deepcopy(X)
        X2[:,1:] *= 2.   # scaled error bars
        X3[:,1:] += 10.  # shifted bars
        X4[:,0] *= 2.    # different predictions
        o = ucc()
        o.fit([X, X2, X3, X4], gt)
        auc1 = o.get_AUUCC(vary_bias=False)
        auc2 = o.get_AUUCC(vary_bias=True)
        assert(len(auc1)==len(auc2)==4 and np.isclose(auc1[0], auc1[1], atol=0.001) and np.isclose(auc2[0], auc2[2], atol=0.001))
        o.set_coordinates(x_axis_name='excess', y_axis_name='deficit', normalize=False)
        auc1 = o.get_AUUCC(vary_bias=False)
        auc2 = o.get_AUUCC(vary_bias=True)
        assert(len(auc1)==len(auc2)==4 and np.isclose(auc1[0], auc1[1], atol=0.001) and np.isclose(auc2[0], auc2[2], atol=0.001))
        op = o.get_OP()
        expop = [(0.45147147547949584, 1.406418455385142, 1.0, 1.0), (1.6309119733785584, 1.0792578573372087, 1.0, 1.0),
                (9.62208333094565, 0.577030310851296, 1.0, 1.0), (0.0, 96.56118140575806, 1.0, 1.0)]
        assert(len(op)==4 and all([np.isclose(a[i],b[i]) for i in range(len(op)) for a,b in zip(op, expop)]))
        # test getting specific OP based on a critical value (scale)
        r1 = o.get_specific_operating_point(req_critical_value=1.0)
        r2 = o.get_specific_operating_point(req_critical_value=2.0)
        assert(r1[1]['new_x']==r2[0]['new_x'])
        assert(r1[1]['new_y']==r2[0]['new_y'])   # because X*2 should be same as X2
        r1 = o.get_specific_operating_point(req_critical_value=0.0, vary_bias=True)
        r2 = o.get_specific_operating_point(req_critical_value=10.0, vary_bias=True)
        assert(r1[2]['new_x']==r2[0]['new_x'])
        assert(r1[2]['new_y']==r2[0]['new_y'])   # because X + 10 should be same as X3


if __name__ == '__main__':
    unittest.main()
