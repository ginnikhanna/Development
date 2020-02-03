import unittest
from CourseraMachineLearning.Utility import linearregression
import numpy as np
from functools import partial

import math


class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_compute_cost_with_regularization_output_for_a_fixed_input_with_zero_regularization(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        X = np.array((1,1)).reshape(2,1)
        y = np.ones(1)
        theta = np.array((1,1)).reshape(2,1)
        lambda_reg = 0

        expected_result = 0.5
        actual_result = linearregression.compute_cost_with_regularization(X, theta, y, lambda_reg)
        self.assertEqual(expected_result, actual_result)

    def test_compute_cost_with_regularization_output_for_a_fixed_input_with_unit_regularization(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        X = np.array((1,1)).reshape(2,1)
        y = np.ones(1)
        theta = np.array((1,1)).reshape(2,1)
        lambda_reg = 1.0

        expected_result = 1
        actual_result = linearregression.compute_cost_with_regularization(X, theta, y, lambda_reg)
        self.assertEqual(expected_result, actual_result)





if __name__ == '__main__':
    unittest.main()
