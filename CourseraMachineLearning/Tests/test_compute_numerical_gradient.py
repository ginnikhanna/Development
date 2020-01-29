import unittest
from CourseraMachineLearning.Utility.neuralnetworks import compute_numerical_gradient, compute_cost, randomly_initialize_parameters, compute_gradients_with_back_propagation
import numpy as np
from functools import partial


class Test(unittest.TestCase):
    """
    Our basic test class
    """
    def test_partial_function(self):

        def add(a : int, b : int):
            return a + b

        add_f = partial(add, b = 2)

        result = add_f(2)

        assert (result == 4)


    def test_that_actual_cost_function_equals_partial_function(self):

        nodes_in = 3
        nodes_out = 5
        number_of_classes = 3
        number_of_training_samples = 5
        EPSILON = 1e-04

        theta_1 = randomly_initialize_parameters(nodes_out, nodes_in + 1)
        theta_2 = randomly_initialize_parameters(number_of_classes, nodes_out + 1)

        X_training = randomly_initialize_parameters(number_of_training_samples, nodes_in)
        y_training = np.array((1,2,3,3,1))
        y_training = y_training.reshape(1,len(y_training))

        X_training = X_training.transpose()

        parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))

        perturb = np.zeros_like((parameters))
        numerical_grad = np.zeros_like((parameters))
        numerical_grad_partial_function = np.zeros_like((parameters))

        cost_func = partial(compute_cost,
                            input_layer_nodes_size= nodes_in,
                            hidden_layer_nodes_size = nodes_out,
                            X = X_training,
                            y = y_training,
                            num_labels = number_of_classes,
                            lambda_for_regularization = 0)

        for index, val in enumerate(perturb):
            perturb[index] = EPSILON
            parameters_plus = parameters + perturb
            parameters_minus = parameters - perturb


            cost_plus = compute_cost(parameters_plus,
                                     nodes_in,
                                     nodes_out, X_training, y_training, number_of_classes,
                                     lambda_for_regularization=0)
            cost_minus = compute_cost(parameters_minus,
                                     nodes_in,
                                     nodes_out, X_training, y_training, number_of_classes,
                                     lambda_for_regularization=0)

            numerical_grad[index] = (cost_plus - cost_minus)/(2 * EPSILON)

            cost_plus_partial_function = cost_func(parameters_plus)
            cost_minus_partial_function = cost_func(parameters_minus)

            numerical_grad_partial_function[index] = (cost_plus_partial_function - cost_minus_partial_function)/(2 * EPSILON)

            perturb[index] = 0


        np.testing.assert_array_equal(numerical_grad, numerical_grad_partial_function)

        print (numerical_grad_partial_function)


    def test_that_output_of_numerical_gradient_descent_is_equal_to_back_propagation_result(self):

        nodes_in = 3
        nodes_out = 5
        number_of_classes = 3
        number_of_training_samples = 5
        EPSILON = 1e-04

        theta_1 = randomly_initialize_parameters(nodes_out, nodes_in + 1)
        theta_2 = randomly_initialize_parameters(number_of_classes, nodes_out + 1)

        theta_1 = np.zeros_like(theta_1)
        theta_2 = np.zeros_like(theta_2)

        X_training = randomly_initialize_parameters(number_of_training_samples, nodes_in)
        y_training = np.array((1, 2, 3, 3, 1))
        y_training = y_training.reshape(1, len(y_training))

        X_training = X_training.transpose()

        parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))

        perturb = np.zeros_like((parameters))
        numerical_grad = np.zeros_like((parameters))
        numerical_grad_partial_function = np.zeros_like((parameters))

        cost_func = partial(compute_cost,
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

            cost_plus_partial_function = cost_func(parameters_plus)
            cost_minus_partial_function = cost_func(parameters_minus)

            numerical_grad_partial_function[index] = (cost_plus_partial_function - cost_minus_partial_function) / (
                        2 * EPSILON)

            perturb[index] = 0


        numerical_grad_back_propagation = compute_gradients_with_back_propagation(parameters,
                                                                                  nodes_in,
                                                                                  nodes_out,
                                                                                  X_training, y_training,
                                                                                  number_of_classes,
                                                                                  lambda_for_regularization=0)

        np.testing.assert_array_almost_equal(numerical_grad_partial_function, numerical_grad_back_propagation)



if __name__ == '__main__':
    unittest.main()