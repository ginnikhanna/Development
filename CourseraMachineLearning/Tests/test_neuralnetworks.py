import unittest
from CourseraMachineLearning.Utility import neuralnetworks
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

        input = 0
        expected_result = 0.5
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result, actual_result)

    def test_sigmoid_with_vector_input_half(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.zeros((3, 1))
        expected_result = np.ones_like(input) * 0.5
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())

    def test_sigmoid_with_matrix_input_half(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.zeros((3, 3))
        expected_result = np.ones_like(input) * 0.5
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())


    def test_sigmoid_with_scalar_input_positive_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = math.inf
        expected_result = 1.0
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result, actual_result)

    def test_sigmoid_with_vector_input_positive_infinity(self):

        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.ones((3, 1)) * math.inf
        expected_result = np.ones_like((input))
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())

    def test_sigmoid_with_matrix_input_positive_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.ones((3, 3)) * math.inf
        expected_result = np.ones_like((input))
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())

    def test_sigmoid_with_scalar_input_negative_infinity(self):

        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = -math.inf
        expected_result = 0.0
        given_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result, given_result)

    def test_sigmoid_with_vector_input_negative_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.ones((3, 1)) * (-math.inf)
        expected_result = np.zeros_like((input))
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())

    def test_sigmoid_with_matrix_input_negative_infinity(self):
        """

        Any method which starts with ``test_`` will considered as a test case.
        """

        input = np.ones((3, 1)) * (-math.inf)
        expected_result = np.zeros_like((input))
        actual_result = neuralnetworks.sigmoid(input)
        self.assertEqual(expected_result.all(), actual_result.all())


    def test_sigmoid_gradient_with_scalar_input_zero(self):

        input = 0
        expected_result = 0.25
        actual_result = neuralnetworks.sigmoid_gradients(input)

        self.assertEqual(expected_result, actual_result)



    # def test_output_for_given_input_for_layer_one(self):
    #
    #     test_input = np.ones((2,1))
    #     expected_output = np.ones((2,1))
    #
    #     actual_output = compute_activation_layer_one(test_input)
    #
    #     np.testing.assert_array_equal(actual_output, expected_output)
    #
    #
    # def test_output_for_given_input_for_layer_two(self):
    #
    #     test_input = np.ones((2,1))
    #     theta = np.ones((2,1)



    #
    # def test_output_for_given_input_parameters_for_3_layer_neural_network(self):
    #
    #     expected_output = np.array((0.5, 0.5))
    #
    #     parameters_input = np.array((0.5, -0.5, -0.5, 1))
    #     X = np.array((1))
    #     X = np.vstack((1, X))
    #
    #     input_layer_nodes_size = 1
    #     hidden_layer_nodes_size = 1
    #     num_labels = 1
    #
    #     parameters_reshaped = reshape_parameters(parameters_input,
    #                                              input_layer_nodes_size,
    #                                              hidden_layer_nodes_size,
    #                                              num_labels)
    #
    #     activation = compute_activation_using_forward_propagation(parameters_reshaped, X)
    #
    #     np.testing.assert_array_almost_equal(expected_output[0], activation[0])
    #     np.testing.assert_array_almost_equal(expected_output[1], activation[1])



if __name__ == '__main__':
    unittest.main()