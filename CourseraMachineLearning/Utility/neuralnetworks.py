from __future__ import division
import numpy as np
from typing import List
from functools import partial


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def sigmoid_gradients(z):
    return sigmoid(z)* (1 - sigmoid(z))


def add_bias_node(X : np.ndarray):
    '''

    :param X: (M x N) matrix with M parameters and N training samples
    :return: (M+1 x N) matrix with 1s added on the 0th row
    '''

    ones = np.ones((1, X.shape[1]))
    X = np.vstack((ones, X))

    return X

def prepare_output_matrix_from_given_output_vector(y : np.ndarray,
                                                   num_labels : int):
    # TODO : This function is very specific to this case of Y. Update to a more general form

    '''

    :param y: output vector with 1 x N dimension
    :param num_labels : number of classes in the output vector
    :return: matrix_( num_labels x N )
    '''

    I = np.eye(num_labels + 1)
    Y = I[y, :]
    Y = Y[0][:, 1:].transpose()

    return Y


def prepare_output_matrix_from_given_output_vector_new(y : np.ndarray,
                                                   num_labels : int):
    # TODO : This function is very specific to this case of Y. Update to a more general form

    '''

    :param y: output vector with 1 x N dimension
    :param num_labels : number of classes in the output vector
    :return: matrix_( num_labels x N )
    '''

    # I = np.eye(num_labels + 1)
    # Y = I[y, :]
    # Y = Y[0][:, 1:].transpose()

    I = np.eye(num_labels)
    Y = I[y,:]
    Y = Y[0].transpose()

    return Y



def compute_activation_layer_one_values(X: np.ndarray) -> np.ndarray :
    '''
    :param X: (M+1) x N with M parameters and N traning samples
    :return:
    '''

    return X

def compute_activation_layer_two_values(theta: np.ndarray,
                                 X: np.ndarray) -> np.ndarray:
    '''

        :param theta: (M+1 x 1)
        :param X: (M+1) x N with M parameters and N traning samples
        :return:
        '''
    assert X.shape[0] == theta.shape[0]

    a = sigmoid(theta.transpose().dot(X))
    ones = np.ones((1, a.shape[1]))
    a = np.vstack((ones, a))

    return a

def compute_activation_layer_three_values(theta: np.ndarray,
                                   a: np.ndarray) -> np.ndarray:
    '''

    :param theta: (M+1 x 1)
    :param a: activation function from previous layer
    :return:
    '''
    assert a.shape[0] == theta.shape[0]

    a_out = sigmoid(theta.transpose().dot(a))

    return a_out


def compute_neural_network_output_with_forward_propagation(theta: List[np.ndarray],
                                                 X: np.ndarray) -> List[np.ndarray]:

    '''

    :param theta: L arrays of (M+1)xN dimension
    :param X: MxN dimension with M parameters and N training samples
    :return:
    '''

    a_1 = compute_activation_layer_one_values(X)
    a_2 = compute_activation_layer_two_values(theta[0], a_1)
    a_3 = compute_activation_layer_three_values(theta[1], a_2)

    return a_3

def reshape_1d_parameter_array_to_respective_two_2d_arrays(parameters: np.ndarray,
                                                           number_of_nodes_input_layer: int,
                                                           number_of_nodes_hidden_layer: int,
                                                           num_labels: int) -> List[np.ndarray]:

    '''

    :param parameters: List[thetas] where thetas are the parameters over all the layers
    :param number_of_nodes_input_layer:
    :param number_of_nodes_hidden_layer:
    :param num_labels:
    :return: List of reshaped thetas [np.ndarray_(number_of_nodes_input_layer + 1 x number_of_nodes_hidden_layer),
                                      np.ndarray_(number_of_nodes_hidden_layer + 1 x num_labels)

    '''
    indices_theta_0 = (number_of_nodes_input_layer + 1) * number_of_nodes_hidden_layer
    indices_theta_1 = indices_theta_0 + (number_of_nodes_hidden_layer + 1) * num_labels

    theta_0 = parameters[:indices_theta_0]
    theta_1 = parameters[indices_theta_0:indices_theta_1]

    assert len(theta_0) + len(theta_1) == len(parameters)

    theta_0 = theta_0.reshape((number_of_nodes_input_layer + 1, number_of_nodes_hidden_layer))
    theta_1 = theta_1.reshape((number_of_nodes_hidden_layer + 1, num_labels))

    return [theta_0, theta_1]



def compute_cost(parameters: np.ndarray, input_layer_nodes_size: int,
                 hidden_layer_nodes_size: int,
                 X: np.ndarray,
                 y: np.ndarray,
                 num_labels: int,
                 lambda_for_regularization: float) -> float:
    '''
    :param theta:  Flat array consisting of all thetas
    :param input_layer_nodes_size
    :param  hidden_layer_nodes_size
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :param lambda_for_regularization
    :return: cost J_theta

    '''
    number_of_training_samples = y.shape[1]
    theta = reshape_1d_parameter_array_to_respective_two_2d_arrays(parameters,
                               input_layer_nodes_size, hidden_layer_nodes_size, num_labels)

    X = add_bias_node(X)
    neural_network_output_after_forward_propagation = compute_neural_network_output_with_forward_propagation(theta, X)

    Y = prepare_output_matrix_from_given_output_vector(y, num_labels)


    cost = sum(-Y * np.log((neural_network_output_after_forward_propagation))
               - (1 - Y) * np.log(1 - neural_network_output_after_forward_propagation))
    cost = sum(cost) / number_of_training_samples

    theta_0_regularization = theta[0][1:, ]
    theta_1_regularization = theta[1][1:, ]

    cost += lambda_for_regularization / (2 * number_of_training_samples) \
               * (sum(sum(theta_0_regularization ** 2)) + sum(sum(theta_1_regularization ** 2)))
    return cost


def randomly_initialize_parameters(nodes_in_incoming_layer, nodes_in_outgoing_layer, rand_f = np.random.rand):
    '''

    :param input_layers: number of nodes in incoming layer
    :param output_layers: number of nodes in outgoing layer
    :return: np.ndarray
    '''
    epsilon_init = np.sqrt(6)/(np.sqrt(nodes_in_incoming_layer + nodes_in_outgoing_layer))
    random_parameters = \
        rand_f(nodes_in_incoming_layer, nodes_in_outgoing_layer) * 2 * epsilon_init - epsilon_init

    return random_parameters


def compute_numerical_gradient(parameters: np.ndarray, input_layer_nodes_size: int,
                 hidden_layer_nodes_size: int,
                 X: np.ndarray,
                 y: np.ndarray,
                 num_labels: int,
                 lambda_reg: float) -> float:
    '''
    :param theta:  Flat array consisting of all thetas
    :param input_layer_nodes_size
    :param  hidden_layer_nodes_size
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :param lambda_for_regularization
    :return:

    '''
    EPSILON = 1e-04

    perturb = np.zeros_like((parameters))
    numerical_gradient = np.zeros_like((parameters))

    cost_func = partial(compute_cost,
                        input_layer_nodes_size=input_layer_nodes_size,
                        hidden_layer_nodes_size=hidden_layer_nodes_size,
                        X=X,
                        y=y,
                        num_labels=num_labels,
                        lambda_for_regularization=lambda_reg)

    for index, val in enumerate(perturb):
        perturb[index] = EPSILON
        parameters_plus = parameters + perturb
        parameters_minus = parameters - perturb


        cost_plus_partial_function = cost_func(parameters_plus)
        cost_minus_partial_function = cost_func(parameters_minus)

        numerical_gradient[index] = (cost_plus_partial_function - cost_minus_partial_function) / (
                    2 * EPSILON)

        perturb[index] = 0

    return numerical_gradient








def compute_gradients_with_back_propagation(parameters: np.ndarray, input_layer_nodes_size: int,
                 hidden_layer_nodes_size: int,
                 X: np.ndarray,
                 y: np.ndarray,
                 num_labels: int,
                 lambda_for_regularization: float) -> float:

    number_of_training_samples = y.shape[1]
    theta = reshape_1d_parameter_array_to_respective_two_2d_arrays(parameters,
                               input_layer_nodes_size,
                               hidden_layer_nodes_size,
                               num_labels)


    Y = prepare_output_matrix_from_given_output_vector(y, num_labels)

    theta_0_grad = np.zeros_like(theta[0])
    theta_1_grad = np.zeros_like(theta[1])

    del_theta_0 = np.zeros_like(theta[0])
    del_theta_1 = np.zeros_like(theta[1])

    for index in range(number_of_training_samples):
        a_1 = X[:,index]
        a_1 = np.hstack((1, a_1))
        z_2 = theta[0].transpose().dot(a_1)
        a_2 = sigmoid(z_2)
        a_2 = np.hstack((1, a_2))
        z_3 = theta[1].transpose().dot(a_2)
        a_3 = sigmoid(z_3)

        d_3 = a_3 - Y[:,index]
        d_2 = d_3.dot(theta[1][1:,].transpose()) * sigmoid_gradients(z_2)

        del_theta_1 += (a_2.reshape(len(a_2), 1)).dot(d_3.reshape(1, len(d_3)))
        del_theta_0 += (a_1.reshape(len(a_1), 1)).dot(d_2.reshape(1, len(d_2)))

    theta[0][0, :] = 0
    theta[1][0, :] = 0


    theta_1_grad = del_theta_1/number_of_training_samples + theta[1] * lambda_for_regularization/number_of_training_samples
    theta_0_grad = del_theta_0/number_of_training_samples + theta[0] * lambda_for_regularization/number_of_training_samples

    theta_grad = np.hstack((theta_0_grad.flatten(), theta_1_grad.flatten()))

    return theta_grad



def predict_outcome_for_digit_dataset(X: np.ndarray, theta: List):
    '''

    :param X: M x N matrix
    :param theta: List[theta_s] where theta_s = M x (number_of_features_in_that_hidden_layer)
    :return: prediction : N x 1 vector with predictions of the output in this case
    '''

    ones = np.ones((1, X.shape[1]))
    X = np.vstack((ones, X))
    assert X.shape[0] == theta[0].shape[0]

    A = sigmoid(theta[0].transpose().dot(X))
    ones = np.ones((1, A.shape[1]))
    A = np.vstack((ones, A))
    assert A.shape[0] == theta[1].shape[0]

    B = sigmoid(theta[1].transpose().dot(A))
    prediction = np.argmax(B, axis=0) + 1

    return prediction


def get_accuracy(prediction: np.ndarray, y: np.ndarray) -> float:
    '''

    :param prediction: predicted vector from optimization N x 1
    :param y: actual output N x 1
    :return: accuracy float
    '''
    accuracy = len(np.where((prediction == y))[0]) / len(y) * 100.0
    return accuracy
















