import numpy as np
import matplotlib.pyplot as plt
from CourseraMachineLearning.Utility import logisticregression

def display_data(X : np.ndarray, number_of_images_to_plot:int) -> plt.figure():
    '''
    Reads a NxM matrix, makes a reshaped matrix and plots the individual images on a
    square grid depending on how many images have to be plotted
    :param X: M x N matrix
    :param : number of images to plot
    N : number of training samples
    M : number of parameters
    :return:
    '''

    random_images = np.random.randint(0, X.shape[0], number_of_images_to_plot)
    rows = np.sqrt(number_of_images_to_plot)
    columns = rows

    number_of_pixels = int(np.sqrt(X.shape[1]))
    fig = plt.figure(1 , figsize=(rows, columns))
    for index, value in enumerate(random_images):
        X_reshaped = X[value].reshape((number_of_pixels,number_of_pixels))
        fig.add_subplot(rows, columns, index + 1)
        plt.imshow(X_reshaped)

    return fig


def one_vs_all(X: np.ndarray, y:np.ndarray, num_labels:int, lambda_for_regularization:float, optimization_algo:str ) -> np.ndarray:
    '''
    :param X: M x N matrix
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
        optimized_parameters = logisticregression.minimize_cost_and_find_theta_with_regularization(theta, X, y_training, lambda_for_regularization, optimization_algo)
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

n    return prediction

    print('End')