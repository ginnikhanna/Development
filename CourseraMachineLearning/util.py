import numpy as np

def compute_univariate_cost_function(X : np.ndarray  , theta : np.ndarray, y:np.ndarray) -> float:

    ''' X : np.array with  2 x N dimensions
        theta : np.array with 2 x 1 dimensions
        y : np.array with  1 x N dimensions
    '''
    J_theta = np.mean((theta.transpose().dot(X) - y) ** 2) / (2)
    return J_theta

def compute_multivariate_cost_function(X : np.ndarray  , theta : np.ndarray, y:np.ndarray) -> float:

    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
    '''


    J_theta = (theta.transpose().dot(X) - y).dot((theta.transpose().dot(X) - y).transpose())/(2*len(y))
    return J_theta


def gradient_descent_univariate (X : np.ndarray  , theta : np.ndarray, y:np.ndarray, number_of_iterations :int, alpha:float) -> tuple:
    '''

    :param X: np.array with  2 x N dimensions
    :param theta: np.array with 2 x 1 dimensions
    :param y: np.array with  1 x N dimensions
    :param number_of_iterations:
    :param alpha: learning rate
    :return:
    '''
    temp = np.zeros_like(theta)

    for n in range(number_of_iterations):
        temp[0] = theta[0] - alpha * np.mean(theta.transpose().dot(X) - y)
        temp[1] = theta[1] - alpha * np.mean((theta.transpose().dot(X) - y) * X.transpose()[:, 1])
        theta = np.array((temp[0], temp[1]))
        J = compute_univariate_cost_function(X, theta, y)

    return theta, J

def gradient_descent_multivariate(X : np.ndarray  , theta : np.ndarray, y:np.ndarray, number_of_iterations :int, alpha:float) -> tuple:
    '''

    :param X: np.array with  M x N dimensions
    :param theta: np.array with M x 1 dimensions
    :param y: data vector with np.array with  1 x N dimensions
    :param number_of_iterations:
    :param alpha: learning rate
    :return:
    '''
    temp = np.zeros_like(theta)

    for n in range(number_of_iterations):

        temp = theta - alpha * X.dot((theta.transpose().dot(X) - y).transpose())/len(y)
        theta = temp
        J = compute_multivariate_cost_function(X, theta, y)
    return theta, J


def normalized_features_matrix(X : np.ndarray) -> np.ndarray:
    '''

    :param X:  np.array with  m x N dimensions , m : number of features, N:number of training samples
    :return: normalized feature Matrix
    '''
    number_of_features = len(X)
    mu = np.zeros((number_of_features, 1))
    sigma = np.zeros((number_of_features, 1))

    mu = X.mean(axis = 1)
    sigma = X.std(axis = 1, ddof=1) # Ddof takes care of division by N-1 and not N to compensate for bias in sample variance

    X = (X.transpose() - mu)/sigma
    X = X.transpose()

    return X, mu, sigma

def parameters_from_normal_equation(X :np.ndarray, y:np.ndarray):
    '''

    :param X: M x N feature matrix
    :param y: 1 x N data vector
    :return:
    '''
    X = X.transpose()
    theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    return theta
