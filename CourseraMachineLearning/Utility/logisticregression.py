import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

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


def minimize_cost_and_find_theta(initial_theta: np.ndarray, X :np.ndarray, y:np.ndarray) -> tuple():

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

def plot_decision_boundary(theta : np.ndarray, X: np.ndarray, y:np.ndarray, fig_number : int) -> plt.figure:

    ''' X : np.array with  M x N dimensions
        theta : np.array with M x 1 dimensions
        y : np.array with  1 x N dimensions
        M : number of parameters
        N : number of training samples
        fig_number : figure number where the features are plotted
        '''

    x_1 = np.array([min(X[1]), max(X[1])])
    x_2 = -theta[0]/theta[2] - theta[1]/theta[2]*x_1
    fig = plt.figure(fig_number)
    plt.plot(x_1, x_2, color = 'k', linewidth = 2, label = 'Decision Boundary')
    plt.legend()
    return fig

def predict_outcome_for_given_dataset(theta: np.ndarray, X : np.ndarray) -> np.ndarray:
    ''' X : np.array with  M x N dimensions
           theta : np.array with M x 1 dimensions
    '''

    probability = sigmoid(X.transpose().dot(theta))
    prediction = (probability > 0.5).astype(int)
    return prediction


def construct_matrix_with_mapped_features(X : np.ndarray, degree) -> np.ndarray:
    '''

    :param X: feature matrix
    :param degree : highest order of the polynomial
    :return: higher dimension feature matrix
    '''

    X_out = np.ones((1, len(X[1])))

    for i in range(1, degree + 1, 1):
        for j in range(0, i + 1, 1):
            X_out = np.vstack((X_out, pow(X[0], (i - j)) * pow(X[1], j)))

    return X_out


