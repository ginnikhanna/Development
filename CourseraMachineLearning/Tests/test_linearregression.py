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
        theta = np.array((1,1))
        lambda_reg = 0

        expected_result = 0.5
        actual_result = linearregression.compute_cost_with_regularization(theta, X, y, lambda_reg)
        self.assertEqual(expected_result, actual_result)

    def test_compute_cost_with_regularization_output_for_a_fixed_input_with_unit_regularization(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        X = np.array((1,1)).reshape(2,1)
        y = np.ones(1)
        theta = np.array((1,1))
        lambda_reg = 1.0

        expected_result = 1
        actual_result = linearregression.compute_cost_with_regularization(theta, X, y, lambda_reg)
        self.assertEqual(expected_result, actual_result)

    def test_compute_gradient_with_regularization_output_for_a_fixed_input_with_zero_regularization(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        X = np.array((1,1)).reshape(2,1)
        y = np.ones(1)
        theta = np.array((1,1))
        lambda_reg = 0

        expected_result = np.ones_like(theta)
        actual_result = linearregression.compute_gradient_with_regularization(theta, X, y, lambda_reg)
        np.testing.assert_array_equal(expected_result, actual_result)

    def test_compute_gradient_with_regularization_output_for_a_fixed_input_with_unit_regularization(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        X = np.array((1,1)).reshape(2,1)
        y = np.ones(1)
        theta = np.array((1,1))
        lambda_reg = 1

        expected_result = np.array((1, 2))
        actual_result = linearregression.compute_gradient_with_regularization(theta, X, y, lambda_reg)
        np.testing.assert_array_equal(expected_result, actual_result)

    def test_get_polynomial_feature_matrix_for_a_given_univariate_feature_matrix(self):

        X = np.array((1, 2, 3, 4)).reshape(1, 4)
        p = 3

        expected_result_1 = np.array((1, 2, 3, 4))
        expected_result_2 = np.array((1, 4, 9, 16))
        expected_result_3 = np.array((1, 8, 27, 64))

        expected_result = np.vstack((expected_result_1, expected_result_2, expected_result_3))
        actual_results = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X, p)

        np.testing.assert_array_equal(expected_result, actual_results)


if __name__ == '__main__':
    unittest.main()
