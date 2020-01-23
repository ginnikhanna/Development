from __future__ import division
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from CourseraMachineLearning.Utility import neuralnetworks
from CourseraMachineLearning.Utility import logisticregression
from CourseraMachineLearning.Utility.logisticregression import OptimizationAlgo
from CourseraMachineLearning.Utility import plot

# Load Data
data = scipy.io.loadmat('ex3data1.mat')
print('End')

X_training = data['X']
y_training = data['y']

#Plot data
plot_training_data = plot.display_data(X_training, 100)

theta = np.array([-2, -1, 1, 2])
X_t = np.arange(1, 16).reshape(3,5)/10
y = np.array([1,0,1,0,1])
ones = np.array(np.ones(5))
X = np.vstack((ones, X_t))

lambda_for_regularization  = 3

cost = logisticregression.compute_cost_with_regularization(theta, X, y, lambda_for_regularization)
gradients = logisticregression.compute_gradients_with_regularization(theta, X, y, lambda_for_regularization)
#gradients = logisticregression.minimize_cost_and_find_theta_with_regularization(theta, X, y, lambda_for_regularization, OptimizationAlgo.MINIMIZE)

print(f'Cost : {cost}')
print(f'Gradients : {gradients}')

num_labels = 10
lambda_for_regularization  = 0.1
all_theta = logisticregression.one_vs_all_classifier(X_training, y_training, num_labels, lambda_for_regularization, OptimizationAlgo.FMIN_CG)

prediction_on_training_set = logisticregression.predict_outcome_for_digit_dataset(X_training, all_theta)
accuracy = logisticregression.get_accuracy(prediction_on_training_set, y_training.flatten())
print(f'Accuracy on the training set is {accuracy}')

