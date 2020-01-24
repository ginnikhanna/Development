import numpy as np
import matplotlib.pyplot as plt
from CourseraMachineLearning.Utility import logisticregression

def display_data(X : np.ndarray, number_of_images_to_plot:int, fig_number:int) -> plt.figure():
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

    if number_of_images_to_plot == 1:
        fig = plt.figure(fig_number, figsize=(3, 3))
        X_reshaped = X.reshape((number_of_pixels,number_of_pixels))
        plt.imshow(X_reshaped)

    elif number_of_images_to_plot > 1:
        fig = plt.figure(fig_number, figsize=(rows, columns))
        for index, value in enumerate(random_images):
            X_reshaped = X[value].reshape((number_of_pixels,number_of_pixels))
            fig.add_subplot(rows, columns, index + 1)
            plt.imshow(X_reshaped)

    return fig


