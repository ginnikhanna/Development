from __future__ import division
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from CourseraMachineLearning.Utility import plot
from CourseraMachineLearning.Utility import neuralnetworks

'''
The neural network is a 3 layer neural network with 
1 input layer, 
1 hidden layer and 
1 output layer
Therefore it will have two stages of thetas or weights 
theta_1 and theta_2 
'''
# Load Data
data = scipy.io.loadmat('ex3data1.mat')

X_training = data['X']
y_training = data['y']

#Plot data
plot_training_data = plot.display_data(X_training,  36, fig_number=1)
#plt.show()

# Load weights of neural network
theta = scipy.io.loadmat('ex3weights.mat')
theta_1 = theta['Theta1']
theta_2 = theta['Theta2']

print(theta_1.shape)
print(theta_2.shape)

number_of_hidden_layers = 2
theta = [theta_1.transpose(), theta_2.transpose()]
prediction = neuralnetworks.predict_outcome_for_digit_dataset(X_training.transpose(), theta)
accuracy = neuralnetworks.get_accuracy(prediction, y_training.flatten())

print(f'Accuracy of the neural network is : {accuracy}')

random_row = np.random.randint(0, 5000)
print(random_row)

input_image = X_training[random_row :random_row + 1,:]
prediction_digit = neuralnetworks.predict_outcome_for_digit_dataset(input_image.transpose(), theta)

print(f'Predicted digit: {prediction_digit} and training digit : {y_training[random_row]}')
plot_random_digit = plot.display_data(input_image, 1, fig_number=2)
plt.show()