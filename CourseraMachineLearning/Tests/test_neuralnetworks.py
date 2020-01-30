import unittest
from CourseraMachineLearning.Utility import neuralnetworks
import numpy as np
from functools import partial

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


    def test_add_bias_node_output_for_a_given_input(self):
        input = np.ones((2,1))
        expected_output = np.ones((3,1))

        actual_output = neuralnetworks.add_bias_node(input)

        np.testing.assert_array_equal(expected_output, actual_output)

    def test_prepare_output_matrix_from_given_output_vector_for_a_given_input(self):

        #TODO : This function is very specific to this case of Y

        input = np.array((1, 2, 3, 4, 1))
        input = input.reshape((1, len(input)))
        num_labels = 4

        expected_output = np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]))
        expected_output = expected_output.transpose()

        actual_output = neuralnetworks.prepare_output_matrix_from_given_output_vector(input, num_labels)

        np.testing.assert_array_equal(expected_output, actual_output)



    def test_compute_activation_layer_one_value_for_a_given_input(self):
        input = np.ones((2,1))
        expected_output = np.ones((2,1))

        actual_output = neuralnetworks.compute_activation_layer_one_values(input)

        np.testing.assert_array_equal(expected_output, actual_output)

    def test_compute_activation_layer_two_value_for_a_given_input(self):
        input = np.zeros((2,1))
        theta = np.ones((2,1))

        expected_output = np.array((1,0.5)).reshape(2,1)

        actual_output = neuralnetworks.compute_activation_layer_two_values(theta, input)

        np.testing.assert_array_equal(expected_output, actual_output)

    def test_compute_activation_layer_three_value_for_a_given_input(self):
        input = np.zeros((2,1))
        theta = np.ones((2,1))

        expected_output = np.array((0.5))

        actual_output = neuralnetworks.compute_activation_layer_three_values(theta, input)

        np.testing.assert_array_equal(expected_output, actual_output)


    def test_compute_neural_network_output_with_forward_propagation_for_a_given_input(self):

        input = np.zeros((2,1))
        theta = [np.ones((2,1)), np.ones((2,1))]

        expected_output = neuralnetworks.sigmoid(np.array((1.5)))

        actual_output = neuralnetworks.compute_neural_network_output_with_forward_propagation(theta, input)

        np.testing.assert_array_equal(expected_output, actual_output)


    def test_reshape_1d_parameter_array_to_respective_two_2d_arrays(self):

        input = np.array((1, 2, 3, 1))
        expected_output = [input[:2].reshape(2,1), input[2:].reshape(2,1)]

        input_layer_nodes_size = 1
        hidden_layer_nodes_size = 1
        num_labels = 1

        parameters_reshaped = neuralnetworks.reshape_1d_parameter_array_to_respective_two_2d_arrays(input.transpose(), input_layer_nodes_size, hidden_layer_nodes_size, num_labels)

        np.testing.assert_array_equal(expected_output[0], parameters_reshaped[0])
        np.testing.assert_array_equal(expected_output[1], parameters_reshaped[1])


    def test_prepare_output_for_different_inputs(self):

        input = np.array((1, 2, 3, 4, 10))
        input = input.reshape(1, len(input))

        input_2 = np.where(input == 10, 0, input)

        num_labels = 10

        output_1 = neuralnetworks.prepare_output_matrix_from_given_output_vector(input, num_labels)

        output_2 = neuralnetworks.prepare_output_matrix_from_given_output_vector_new(input_2, num_labels)



    def fake_matrix_rand_f(self, rows, columns):
        return np.ones((rows, columns)) / 3


    def test_that_output_is_random_matrix_with_all_values_non_zero(self):
        nodes_in_incoming_layer = 3
        nodes_in_outgoing_layer = 2

        given_results = np.zeros((nodes_in_incoming_layer,nodes_in_outgoing_layer))

        R = neuralnetworks.randomly_initialize_parameters(nodes_in_incoming_layer, nodes_in_outgoing_layer)

        self.assertNotEqual(given_results.all(), R.all())


    def test_that_the_output_of_numerical_gradient_is_equal_for_both_direct_and_partial_function_implementation(self):
        nodes_in = 3
        nodes_out = 5
        number_of_classes = 3
        number_of_training_samples = 5
        EPSILON = 1e-04

        theta_1 = neuralnetworks.randomly_initialize_parameters(nodes_out, nodes_in + 1)
        theta_2 = neuralnetworks.randomly_initialize_parameters(number_of_classes, nodes_out + 1)

        X_training = neuralnetworks.randomly_initialize_parameters(number_of_training_samples, nodes_in)
        y_training = np.array((1, 2, 3, 3, 1))
        y_training = y_training.reshape(1, len(y_training))

        X_training = X_training.transpose()

        parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))

        perturb = np.zeros_like((parameters))
        numerical_grad = np.zeros_like((parameters))
        numerical_grad_partial_function = np.zeros_like((parameters))

        cost_func = partial(neuralnetworks.compute_cost,
                            input_layer_nodes_size=nodes_in,
                            hidden_layer_nodes_size=nodes_out,
                            X=X_training,
                            y=y_training,
                            num_labels=number_of_classes,
                            lambda_for_regularization=0)

        for index, val in enumerate(perturb):
            perturb[index] = EPSILON
            parameters_plus = parameters + perturb
            parameters_minus = parameters - perturb

            cost_plus = neuralnetworks.compute_cost(parameters_plus,
                                     nodes_in,
                                     nodes_out, X_training, y_training, number_of_classes,
                                     lambda_for_regularization=0)
            cost_minus = neuralnetworks.compute_cost(parameters_minus,
                                      nodes_in,
                                      nodes_out, X_training, y_training, number_of_classes,
                                      lambda_for_regularization=0)

            numerical_grad[index] = (cost_plus - cost_minus) / (2 * EPSILON)

            cost_plus_partial_function = cost_func(parameters_plus)
            cost_minus_partial_function = cost_func(parameters_minus)

            numerical_grad_partial_function[index] = (cost_plus_partial_function - cost_minus_partial_function) / (
                        2 * EPSILON)

            perturb[index] = 0

        np.testing.assert_array_equal(numerical_grad, numerical_grad_partial_function)


    def test_that_numerical_gradient_output_is_equal_to_gradient_obtained_from_back_propagation(self):
        nodes_in = 1
        nodes_out = 1
        number_of_classes = 2
        number_of_training_samples = 5
        EPSILON = 1e-04
        lambda_reg = 3

       # theta_1 = neuralnetworks.randomly_initialize_parameters(nodes_out, nodes_in + 1)
       # theta_2 = neuralnetworks.randomly_initialize_parameters(number_of_classes, nodes_out + 1)

        theta_1 = np.ones((nodes_out, nodes_in + 1))
        theta_2 = np.ones((number_of_classes, nodes_out + 1))

  #      X_training = neuralnetworks.randomly_initialize_parameters(number_of_training_samples, nodes_in)
        X_training = np.ones((number_of_training_samples, nodes_in)) * 2
        y_training = np.array((1, 2, 2, 2, 1))
        y_training = y_training.reshape(1, len(y_training))

        X_training = X_training.transpose()

        parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))


        numerical_grad = neuralnetworks.compute_numerical_gradient(parameters,
                                                                   nodes_in,
                                                                   nodes_out,
                                                                   X_training,
                                                                   y_training,
                                                                   number_of_classes,
                                                                   lambda_reg)
        back_propagation_grad = neuralnetworks.compute_gradients_with_back_propagation(parameters,
                                                                                       nodes_in,
                                                                                       nodes_out, X_training, y_training,
                                                                                       number_of_classes, lambda_reg)

        np.testing.assert_array_almost_equal(numerical_grad, back_propagation_grad)




if __name__ == '__main__':
    unittest.main()