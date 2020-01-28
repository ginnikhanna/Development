from __future__ import division
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from CourseraMachineLearning.Utility import plot
from CourseraMachineLearning.Utility import neuralnetworks

'''
A backpropagation algorithm is implemented to classify the digits 
The neural network consists of 3 layers 
1 Input layer, 1 hidden layer and 1 output layer 
The input consists of feature size of 400 elements. So the first layer has 400 nodes. 
'''
# Load Data
num_of_classes = 10
input_layer_nodes_size = 400
hidden_layer_nodes_size = 25

data = scipy.io.loadmat('ex4data1.mat')
X_training = data['X']
y_training = data['y']


#Plot data
plot_training_data = plot.display_data(X_training,  36, fig_number=1)
plt.show()


# Load weights of neural network
theta = scipy.io.loadmat('ex4weights.mat')
theta_1 = theta['Theta1']
theta_2 = theta['Theta2']

parameters = np.hstack((theta_1.transpose().flatten(), theta_2.transpose().flatten()))
theta = [theta_1.transpose(), theta_2.transpose()]


X_training = X_training.transpose()
y_training = y_training.transpose()

cost = neuralnetworks.compute_cost(parameters,
                                   input_layer_nodes_size,
                                   hidden_layer_nodes_size,
                                   X_training, y_training, num_of_classes)
print(f'Cost :{cost}')