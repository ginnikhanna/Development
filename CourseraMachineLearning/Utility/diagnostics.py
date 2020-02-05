import numpy as np
from CourseraMachineLearning.Utility import linearregression
from typing import List
def get_training_error(X_training : np.ndarray,
                       y_training : np.ndarray,
                       X_crossval : np.ndarray,
                       y_crossval : np.ndarray,
                       lambda_for_regularization):

    '''

    :param X: np.ndarray with M x N dimension
    :param y:
    :return:
    '''

    number_of_samples = X_training.shape[1]
    initial_theta = np.zeros((X_training.shape[0]))
    lambda_for_reg = lambda_for_regularization

    error_training = []
    error_cross_val = []

    for i in range(number_of_samples):

        X_training_i = X_training[:,:i + 1]
        y_training_i = y_training[:i + 1]
        optimized_theta  = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                                             X_training_i,
                                                                                             y_training_i,
                                                                                             lambda_for_reg,
                                                                                             linearregression.OptimizationAlgo.MINIMIZE)

        error_training.append(np.mean((optimized_theta.dot(X_training_i) - y_training_i)**2)/2)
        error_cross_val.append(np.mean((optimized_theta.dot(X_crossval) - y_crossval)**2)/2)

    return error_training, error_cross_val


def get_validation_error(X_training : np.ndarray,
                       y_training : np.ndarray,
                       X_crossval : np.ndarray,
                       y_crossval : np.ndarray,
                       lambda_for_regularization : List, ):

    '''
    Validation error is calculated over the range of lambda.
    Optimization of theta wrt different lambdas
    :param X: np.ndarray with M x N dimension
    :param y: N x 1
    :return:
    '''

    initial_theta = np.zeros((X_training.shape[0]))

    error_training = []
    error_cross_val = []

    for lambda_i in lambda_for_regularization:

        optimized_theta  = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                                             X_training,
                                                                                             y_training,
                                                                                             lambda_i,
                                                                                             linearregression.OptimizationAlgo.FMIN_CG)

        error_training.append(np.mean((optimized_theta.dot(X_training) - y_training)**2)/2)
        error_cross_val.append(np.mean((optimized_theta.dot(X_crossval) - y_crossval)**2)/2)

    return error_training, error_cross_val