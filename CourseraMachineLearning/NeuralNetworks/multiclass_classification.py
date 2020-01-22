from __future__ import division
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from CourseraMachineLearning.Utility import neuralnetworks
from CourseraMachineLearning.Utility import logisticregression
# Load Data
data = scipy.io.loadmat('ex3data1.mat')
print('End')

X_training = data['X']
y_training = data['y']

#Plot data
plot_training_data = neuralnetworks.display_data(X_training, 100)

theta = np.array([-2, -1, 1, 2])
X_t = np.arange(1, 16).reshape(3,5)/10
y = np.array([1,0,1,0,1])
ones = np.array(np.ones(5))

X = np.vstack((ones, X_t))
cost = logisticregression.compute_cost_with_regularization(theta, X, y, lambda_for_regularization=3)
gradients = logisticregression.compute_gradients_with_regularization(theta, X, y, lambda_for_regularization=3)

gradients = logisticregression.minimize_cost_and_find_theta_with_regularization(theta, X, y, lambda_for_regularization=3)

print(f'Cost : {cost}')
print(f'Gradients : {gradients}')
#
num_labels = 10
neuralnetworks.one_vs_all(X_training, y_training, num_labels)

#plt.show()
