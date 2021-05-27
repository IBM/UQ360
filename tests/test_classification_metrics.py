import unittest
from unittest import TestCase
import numpy as np


class TestExpectedCalibrationError(TestCase):

    def test_expected_calibration_error_on_calibrated_preditions(self):
        from uq360.metrics import expected_calibration_error

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.25, 0.75], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]])
        ece = expected_calibration_error(y_true, y_prob)

        assert np.isclose(ece, 0.0), "ece is: {:.2f}".format(ece)

    def test_expected_calibration_error_on_calibrated_sharp_preditions(self):
        from uq360.metrics import expected_calibration_error

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        ece = expected_calibration_error(y_true, y_prob)

        assert np.isclose(ece, 0.0), "ece is: {:.2f}".format(ece)

    def test_expected_calibration_error_on_miscalibrated_preditions(self):
        from uq360.metrics import expected_calibration_error

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]])
        ece = expected_calibration_error(y_true, y_prob)

        assert np.isclose(ece, 0.25), "ece is: {:.2f}".format(ece)

    def test_expected_calibration_error_on_miscalibrated_sharp_preditions(self):
        from uq360.metrics import expected_calibration_error

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        ece = expected_calibration_error(y_true, y_prob)

        assert np.isclose(ece, 1.0), "ece is: {:.2f}".format(ece)


if __name__ == '__main__':
    unittest.main()
