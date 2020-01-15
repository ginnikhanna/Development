import numpy as np

def sigmoid(z):

    sig = 1/(1 + np.exp(z))
    return sig

def compute_cost(X : np.ndarray  , theta : np.ndarray, y:np.ndarray) -> float:
    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
    '''

    J_theta = np.mean(-y * np.log(sigmoid(theta.transpose().dot(X))) - (1-y) * np.log(1 - sigmoid(theta.transpose().dot(X))))
    return J_theta

def compute_gradients(X : np.ndarray  , theta : np.ndarray, y:np.ndarray) -> np.ndarray:
    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
    '''

    gradients = (sigmoid(theta.transpose().dot(X)) - y).dot(X.transpose())/len(y)


    return gradients