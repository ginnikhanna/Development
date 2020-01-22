import numpy as np
import matplotlib.pyplot as plt

def display_data(X : np.ndarray, number_of_images_to_plot:int) -> plt.figure():
    '''

    :param X: N x M matrix
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