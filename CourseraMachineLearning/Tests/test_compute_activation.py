import unittest
from CourseraMachineLearning.Utility.neuralnetworks import compute_activation_using_forward_propagation, reshape_parameters
import numpy as np

import math

class Test(unittest.TestCase):
    """
    Our basic test class
    """

    def test_compute_activation_using_forward_propagation(self):

        given_output = np.array((0.5, 0.5))

        parameters = np.array((0.5, -0.5, -0.5, 1))
        X = np.array((1))
        X = np.vstack((1, X))

        input_layer_nodes_size = 1
        hidden_layer_nodes_size = 1
        num_labels = 1

        parameters_reshaped = reshape_parameters(parameters,
                                                 input_layer_nodes_size,
                                                 hidden_layer_nodes_size,
                                                 num_labels)

        activation = compute_activation_using_forward_propagation(parameters_reshaped, X)
        print(activation)

        np.testing.assert_array_almost_equal(given_output[0], activation[0])
        np.testing.assert_array_almost_equal(given_output[1], activation[1])

if __name__ == '__main__':
    unittest.main()