import numpy as np
from CourseraMachineLearning.Utility import linearregression

def get_training_error(X_training : np.ndarray,
                       y_training : np.ndarray,
                       X_crossval : np.ndarray,
                       y_crossval : np.ndarray):

    '''

    :param X: np.ndarray with M x N dimension
    :param y:
    :return:
    '''

    number_of_samples = X_training.shape[1]
    initial_theta = np.array((0,0))
    lambda_for_reg = 0

    error_training = []
    error_cross_val = []

    for i in range(number_of_samples):

        X_training_i = X_training[:,:i + 1]
        y_training_i = y_training[:,:i + 1]
        optimized_theta  = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                                             X_training_i,
                                                                                             y_training_i,
                                                                                             lambda_for_reg,
                                                                                             linearregression.OptimizationAlgo.MINIMIZE)

        error_training.append(np.mean((optimized_theta.dot(X_training_i) - y_training_i)**2)/2)
        error_cross_val.append(np.mean((optimized_theta.dot(X_crossval) - y_crossval)**2)/2)

    return error_training, error_cross_val