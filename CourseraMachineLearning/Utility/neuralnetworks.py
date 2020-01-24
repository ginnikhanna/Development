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

