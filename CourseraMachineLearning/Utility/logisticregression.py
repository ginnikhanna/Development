import numpy as np
import scipy.optimize as so

def sigmoid(z):

    sig = 1/(1 + np.exp(-z))
    return sig

def compute_cost(theta : np.ndarray, X : np.ndarray , y:np.ndarray) -> float:
    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
        NOTE : the first argument is a variable which has to be optimized later with the minimization algorithm
    '''

    J_theta = np.mean(-y * np.log(sigmoid(theta.transpose().dot(X))) - (1-y) * np.log(1 - sigmoid(theta.transpose().dot(X))))

    return J_theta

def compute_gradients(theta : np.ndarray, X : np.ndarray , y:np.ndarray) -> np.ndarray:
    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
       NOTE : the first argument is a variable which has to be optimized later with the minimization algorithm
    '''
    gradients = (sigmoid(theta.transpose().dot(X)) - y).dot(X.transpose())/len(y)
    return gradients


def minimize_cost_and_find_theta(initial_theta, X, y):

    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
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

