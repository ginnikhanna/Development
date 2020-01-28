import unittest
from CourseraMachineLearning.Utility.neuralnetworks import sigmoid_gradients
import numpy as np
import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_sigmoid_with_scalar_input_zero(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input_scalar = 0
        given_res_scalar = 0.25
        test_res_scalar = sigmoid_gradients(input_scalar)
        self.assertEqual(given_res_scalar, test_res_scalar)



if __name__ == '__main__':
    unittest.main()