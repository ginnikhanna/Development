import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from enum import Enum


class OptimizationAlgo(Enum):
    MINIMIZE = 1
    FMIN_CG = 2


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig


def compute_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :return: cost J_theta
    '''
    J_theta = np.mean(
        -y * np.log(sigmoid(theta.transpose().dot(X))) - (1 - y) * np.log(1 - sigmoid(theta.transpose().dot(X))))

    return J_theta


def compute_cost_with_regularization(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                                     lambda_for_regularization: float) -> float:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :param lambda_for_regularization : float
    :return: cost J_theta
    '''
    theta_for_regularization = theta[1:]
    J_theta = np.mean(-y * np.log(sigmoid(theta.transpose().dot(X))) -
                      (1 - y) * np.log(1 - sigmoid(theta.transpose().dot(X)))) \
              + lambda_for_regularization / (2 * len(y)) * np.sum(theta_for_regularization ** 2)

    return J_theta


def compute_gradients(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''

        :param theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :return: gradients
        '''
    gradients = (sigmoid(theta.transpose().dot(X)) - y).dot(X.transpose()) / len(y)
    return gradients


def compute_gradients_with_regularization(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                                          lambda_for_regularization: float) -> np.ndarray:
    '''

        :param theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :return: gradients
        '''

    theta_t = np.hstack((0, theta[1:]))
    gradients = (sigmoid(theta.transpose().dot(X)) - y).dot(X.transpose()) / len(y) + lambda_for_regularization / len(
        y) * theta_t
    return gradients


def minimize_cost_and_find_theta(initial_theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple():
    '''
       :param initial_theta: np.array with Mx1 dimension
       :param X: np.array MxN dimension
       :param y: np.array with 1xN dimension
       :return: optimized parameters thetas
    '''

    # Advanced minimizing algorithm
    result = so.minimize(fun=compute_cost,
                         x0=initial_theta,
                         args=(X, y),
                         jac=compute_gradients)
    '''
    fun : function to minimize, in this case it is compute_cost 
    x0 : initial value of the variable to be optimized for minimum cost 
    args : additional arguments to the compute_cost function 
    jac : function to calculate the gradient 
    '''
    return result


def minimize_cost_and_find_theta_with_regularization(initial_theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                                                     lambda_for_regularization, algo: OptimizationAlgo) -> tuple():
    '''
       :param initial_theta: np.array with Mx1 dimension
       :param X: np.array MxN dimension
       :param y: np.array with 1xN dimension
       :param lambda_for_regularization
       :return: optimized parameters thetas
    '''

    if algo == OptimizationAlgo.MINIMIZE:
        # Advanced minimizing algorithm
        result = so.minimize(fun=compute_cost_with_regularization,
                             x0=initial_theta,
                             args=(X, y, lambda_for_regularization),
                             jac=compute_gradients_with_regularization)
        result = result.x

    if algo == OptimizationAlgo.FMIN_CG:
        result = so.fmin_cg(f = compute_cost_with_regularization,
                            x0 = initial_theta,
                            fprime=compute_gradients_with_regularization,
                            args=(X, y, lambda_for_regularization),
                            maxiter=50)

    '''
    fun : function to minimize, in this case it is compute_cost 
    x0 : initial value of the variable to be optimized for minimum cost 
    args : additional arguments to the compute_cost function 
    jac : function to calculate the gradient 
    '''
    return result


def plot_decision_boundary_line(theta: np.ndarray, X: np.ndarray, y: np.ndarray, fig_number: int) -> plt.figure:
    '''
        :param initial_theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :param fig_number
        :return: figure object
        '''

    if theta.shape[0] <= 2:
        x_1 = np.array([min(X[1]), max(X[1])])
        x_2 = -theta[0] / theta[2] - theta[1] / theta[2] * x_1
        fig = plt.figure(fig_number)
        plt.plot(x_1, x_2, color='k', linewidth=2, label='Decision Boundary')
        plt.legend()
        return fig


def plot_decision_boundary_contours(theta: np.ndarray,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    fig_number: int,
                                    color: str) -> plt.figure:
    '''
        :param initial_theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :param fig_number
        :return: figure object
    '''

    features_min = X.min(axis=1)
    features_max = X.max(axis=1)

    u = np.linspace(features_min[0], features_max[0], 50)
    v = np.linspace(features_min[0], features_max[0], 50)

    z = np.zeros((len(u), len(v)))

    for i, val_u in enumerate(u):
        for j, val_v in enumerate(v):
            input_feature_matrix = np.vstack((val_u, val_v))
            input_mapped_feature_matrix = construct_matrix_with_mapped_features(input_feature_matrix, degree=6)
            z[i, j] = input_mapped_feature_matrix.transpose().dot(theta)

    fig = plt.figure(fig_number)
    cntr = plt.contour(u, v, z.transpose(), levels=0, colors=color)
    return cntr


def predict_outcome_for_given_dataset(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :return: prediction for the dataset
    '''
    probability = sigmoid(X.transpose().dot(theta))
    prediction = (probability > 0.5).astype(int)

    return prediction


def get_accuracy(prediction: np.ndarray, y: np.ndarray) -> float:
    '''

    :param prediction: predicted vector from optimization
    :param y: actual output
    :return: accuracy
    '''
    accuracy = len(np.where((prediction == y))[0]) / len(y) * 100.0
    return accuracy


def construct_matrix_with_mapped_features(X: np.ndarray, degree) -> np.ndarray:
    '''

    :param X: feature matrix
    :param degree : highest order of the polynomial
    :return: higher dimension feature matrix X_out
    '''

    X_out = np.ones((1, len(X[1])))

    for i in range(1, degree + 1, 1):
        for j in range(0, i + 1, 1):
            X_out = np.vstack((X_out, pow(X[0], (i - j)) * pow(X[1], j)))

    return X_out

def one_vs_all_classifier(X: np.ndarray, y:np.ndarray, num_labels:int, lambda_for_regularization:float, optimization_algo:str ) -> np.ndarray:
    '''
    :param X: N x M matrix
    :param y: 1 x N vector
    :param theta : M x 1 vector
    :param lambda_for_regularization
    :param optimization_algo : 'minimize', 'fmin_cg'
    M : number of parameteres
    N : number of training samples
    :return: optimized_theta with dimensions
    '''

    ones = np.ones((1, X.shape[0]))
    X = X.transpose()
    X = np.vstack((ones, X))
    y = y.flatten()

    theta = np.zeros((X.shape[0]))
    optimized_theta = np.zeros_like(theta)
    all_theta = np.zeros((X.shape[0], num_labels))
    cost = np.zeros((num_labels))

    for label in np.arange(1, 11, 1):
        y_training = (y == label).astype(int)
        optimized_parameters = minimize_cost_and_find_theta_with_regularization(theta, X, y_training, lambda_for_regularization, optimization_algo)
        all_theta[:,label-1] = optimized_parameters

    return all_theta

def predict_outcome_for_digit_dataset(X : np.ndarray, theta : np.ndarray):
    '''

    :param X: M x N matrix
    :param theta: M x number_of_digits
    :return: prediction : 1 x N
    '''
    ones = np.ones((1, X.shape[0]))
    X = X.transpose()
    X = np.vstack((ones, X))
    prediction = X.transpose().dot(theta)
    prediction = np.argmax(prediction, axis = 1) + 1

    return prediction
