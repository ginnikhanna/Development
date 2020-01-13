import numpy as np

def compute_univariate_cost_function(X : np.ndarray  , theta : np.ndarray, y:np.ndarray) -> float:

    ''' X : np.array with  2 x N dimensions
        theta : np.array with 2 x 1 dimensions
        y : np.array with  1 x N dimensions
    '''
    J_theta = np.mean((theta.transpose().dot(X) - y) ** 2) / (2)
    return J_theta


def gradient_descent_univariate (X : np.ndarray  , theta : np.ndarray, y:np.ndarray, number_of_iterations :int, alpha:float) -> tuple:
    temp = np.zeros_like(theta)

    for n in range(number_of_iterations):
        temp[0] = theta[0] - alpha * np.mean(theta.transpose().dot(X) - y)
        temp[1] = theta[1] - alpha * np.mean((theta.transpose().dot(X) - y) * X.transpose()[:, 1])
        theta = np.array((temp[0], temp[1]))
        J = compute_univariate_cost_function(X, theta, y)

    return theta, J