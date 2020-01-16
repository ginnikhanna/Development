import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

def sigmoid(z):

    sig = 1/(1 + np.exp(-z))
    return sig

def compute_cost(theta : np.ndarray, X : np.ndarray , y:np.ndarray) -> float:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :return: cost J_theta
    '''
    J_theta = np.mean(-y * np.log(sigmoid(theta.transpose().dot(X))) - (1-y) * np.log(1 - sigmoid(theta.transpose().dot(X))))

    return J_theta

def compute_cost_with_regularization(theta : np.ndarray, X : np.ndarray , y:np.ndarray, lambda_for_regularization :float) -> float:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :param y: np.array with 1xN dimension
    :param lambda_for_regularization : float
    :return: cost J_theta
    '''
    theta_for_regularization = theta[1:]
    J_theta = np.mean(-y * np.log(sigmoid(theta.transpose().dot(X))) -
                      (1-y) * np.log(1 - sigmoid(theta.transpose().dot(X)))) \
                       + lambda_for_regularization/(2*len(y)) * np.sum(theta_for_regularization**2)

    return J_theta


def compute_gradients(theta : np.ndarray, X : np.ndarray , y:np.ndarray) -> np.ndarray:
    '''

        :param theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :return: gradients
        '''
    gradients = (sigmoid(theta.transpose().dot(X)) - y).dot(X.transpose())/len(y)
    return gradients


def minimize_cost_and_find_theta(initial_theta: np.ndarray, X :np.ndarray, y:np.ndarray) -> tuple():
    '''
       :param initial_theta: np.array with Mx1 dimension
       :param X: np.array MxN dimension
       :param y: np.array with 1xN dimension
       :return: optimized parameters thetas
    '''

    # Advanced minimizing algorithm
    result = so.minimize(fun =compute_cost,
                         x0 =initial_theta,
                         args = (X,y),
                         jac = compute_gradients)
    '''
    fun : function to minimize, in this case it is compute_cost 
    x0 : initial value of the variable to be optimized for minimum cost 
    args : additional arguments to the compute_cost function 
    jac : function to calculate the gradient 
    '''
    return result

def plot_decision_boundary(theta : np.ndarray, X: np.ndarray, y:np.ndarray, fig_number : int) -> plt.figure:
    '''
        :param initial_theta: np.array with Mx1 dimension
        :param X: np.array MxN dimension
        :param y: np.array with 1xN dimension
        :param fig_number
        :return: figure object
        '''

    x_1 = np.array([min(X[1]), max(X[1])])
    x_2 = -theta[0]/theta[2] - theta[1]/theta[2]*x_1
    fig = plt.figure(fig_number)
    plt.plot(x_1, x_2, color = 'k', linewidth = 2, label = 'Decision Boundary')
    plt.legend()
    return fig

def predict_outcome_for_given_dataset(theta: np.ndarray, X : np.ndarray) -> np.ndarray:
    '''

    :param theta: np.array with Mx1 dimension
    :param X: np.array MxN dimension
    :return: prediction for the dataset
    '''
    probability = sigmoid(X.transpose().dot(theta))
    prediction = (probability > 0.5).astype(int)
    return prediction


def construct_matrix_with_mapped_features(X : np.ndarray, degree) -> np.ndarray:
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


