import numpy as np
from typing import List

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def predict_outcome_for_digit_dataset(X : np.ndarray, theta : List):
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
    prediction = np.argmax(B, axis = 0) + 1

    return prediction

def get_accuracy(prediction: np.ndarray, y: np.ndarray) -> float:
    '''

    :param prediction: predicted vector from optimization N x 1
    :param y: actual output N x 1
    :return: accuracy float
    '''
    accuracy = len(np.where((prediction == y))[0]) / len(y) * 100.0
    return accuracy


def reshape_parameters(parameters:np.ndarray,
                       input_layer_nodes_size:int,
                       hidden_layer_nodes_size : int,
                       num_labels: int) -> List[np.ndarray] :


    indices_theta_0 = (input_layer_nodes_size + 1) * hidden_layer_nodes_size
    indices_theta_1 = indices_theta_0 + (hidden_layer_nodes_size + 1) * num_labels

    theta_0 = parameters[:indices_theta_0]
    theta_1 = parameters[indices_theta_0:indices_theta_1]

    assert len(theta_0) + len(theta_1) == len(parameters)

    theta_0 = theta_0.reshape((input_layer_nodes_size + 1, hidden_layer_nodes_size))
    theta_1 = theta_1.reshape((hidden_layer_nodes_size + 1 , num_labels))

    return [theta_0, theta_1]

def compute_cost(parameters: np.ndarray, input_layer_nodes_size:int,
                 hidden_layer_nodes_size : int,
                 X: np.ndarray,
                 y: np.ndarray,
                 num_labels : int) -> float:

    '''

    :param theta:  Flat array consisting of all thetas
    :param input_layer_nodes_size
    :param  hidden_layer_nodes_size
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :return: cost J_theta
    '''
    number_of_training_samples = y.shape[1]
    theta = reshape_parameters(parameters,
                               input_layer_nodes_size, hidden_layer_nodes_size, num_labels)

    ones = np.ones((1, X.shape[1]))
    X = np.vstack((ones, X))
    assert X.shape[0] == theta[0].shape[0]

    A_2 = sigmoid(theta[0].transpose().dot(X))
    ones = np.ones((1, A_2.shape[1]))
    A_2 = np.vstack((ones, A_2))
    assert A_2.shape[0] == theta[1].shape[0]

    A_3 = sigmoid(theta[1].transpose().dot(A_2))

    I = np.eye(num_labels + 1)
    Y = I[y, :]
    Y = Y[0][:,1:].transpose()

    cost = sum(-Y * np.log((A_3)) - (1-Y) * np.log(1-A_3))
    J_theta = sum(cost)/number_of_training_samples
    return J_theta
