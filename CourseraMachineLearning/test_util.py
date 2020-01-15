import unittest
from CourseraMachineLearning.util import sigmoid
import numpy as np

class TestFactorial(unittest.TestCase):
    """
    Our basic test class
    """

    def test_sigmoid(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """

        input_scalar = 0
        input_vector = np.zeros((3,1))
        input_matrix = np.zeros((3,3))

        given_res_scalar = 0.5
        given_res_vector = np.ones_like(input_vector) * 0.5
        given_res_matrix = np.ones_like(input_matrix) * 0.5

        test_res_scalar = sigmoid(0)
        test_res_vector = sigmoid(input_vector)
        test_res_matrix = sigmoid(input_matrix)



        self.assertEqual(given_res_scalar, 0.5)
        self.assertEqual(given_res_vector.all(), test_res_vector.all())
        self.assertEqual(given_res_matrix.all(), test_res_matrix.all())

if __name__ == '__main__':
    unittest.main()