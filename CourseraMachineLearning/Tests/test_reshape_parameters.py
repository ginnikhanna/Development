import unittest
from CourseraMachineLearning.Utility.neuralnetworks import reshape_parameters
import numpy as np

import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_reshape_parameters(self):

        parameters = np.array((1, 2, 3, 1))
        given_output = [parameters[:2].reshape(2,1), parameters[2:].reshape(2,1)]

        input_layer_nodes_size = 1
        hidden_layer_nodes_size = 1
        num_labels = 1

        parameters_reshaped = reshape_parameters(parameters.transpose(), input_layer_nodes_size, hidden_layer_nodes_size, num_labels)

        np.testing.assert_array_equal(given_output[0], parameters_reshaped[0])
        np.testing.assert_array_equal(given_output[1], parameters_reshaped[1])





if __name__ == '__main__':
    unittest.main()