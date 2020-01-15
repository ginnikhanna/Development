import unittest
from CourseraMachineLearning.Utility.logisticregression import sigmoid
import numpy as np
import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_sigmoid_with_scalar_input_half(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_scalar = 0
        given_res_scalar = 0.5
        test_res_scalar = sigmoid(input_scalar)
        self.assertEqual(given_res_scalar, test_res_scalar)

    def test_sigmoid_with_vector_input_half(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_vector = np.zeros((3, 1))
        given_res_vector = np.ones_like(input_vector) * 0.5
        test_res_vector = sigmoid(input_vector)
        self.assertEqual(given_res_vector.all(), test_res_vector.all())

    def test_sigmoid_with_matrix_input_half(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_matrix = np.zeros((3, 3))
        given_res_matrix = np.ones_like(input_matrix) * 0.5
        test_res_matrix = sigmoid(input_matrix)
        self.assertEqual(given_res_matrix.all(), test_res_matrix.all())


    def test_sigmoid_with_scalar_input_positive_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_scalar = math.inf
        given_res_scalar = 0.0
        test_res_scalar = sigmoid(input_scalar)
        self.assertEqual(given_res_scalar, test_res_scalar)

    def test_sigmoid_with_vector_input_positive_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_vector = np.ones((3, 1)) * math.inf
        given_res_vector = np.zeros_like((input_vector))
        test_res_vector = sigmoid(input_vector)
        self.assertEqual(given_res_vector.all(), test_res_vector.all())

    def test_sigmoid_with_matrix_input_positive_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_matrix = np.ones((3, 3)) * math.inf
        given_res_matrix = np.zeros_like((input_matrix))
        test_res_matrix = sigmoid(input_matrix)
        self.assertEqual(given_res_matrix.all(), test_res_matrix.all())

    def test_sigmoid_with_scalar_input_negative_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_scalar = -math.inf
        given_res_scalar = 1.0
        test_res_scalar = sigmoid(input_scalar)
        self.assertEqual(given_res_scalar, test_res_scalar)

    def test_sigmoid_with_vector_input_negative_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_vector = np.ones((3, 1)) * (-math.inf)
        given_res_vector = np.ones_like((input_vector))
        test_res_vector = sigmoid(input_vector)
        self.assertEqual(given_res_vector.all(), test_res_vector.all())

    def test_sigmoid_with_matrix_input_negative_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_matrix = np.ones((3, 1)) * (-math.inf)
        given_res_matrix = np.ones_like((input_matrix))
        test_res_matrix = sigmoid(input_matrix)
        self.assertEqual(given_res_matrix.all(), test_res_matrix.all())



if __name__ == '__main__':
    unittest.main()