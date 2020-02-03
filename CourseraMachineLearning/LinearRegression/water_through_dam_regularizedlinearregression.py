import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io

import pandas as pd
from CourseraMachineLearning.Utility import linearregression
from CourseraMachineLearning.Utility.linearregression import OptimizationAlgo

#Load Data
data = scipy.io.loadmat('ex5data1.mat')

X_training = data['X']
X_val = data['Xval']
X_test = data['Xtest']

y_training = data['y']
y_val = data['yval']
y_test = data['ytest']

# Plot Data

plt.figure(1)
plt.scatter(X_training, y_training)
plt.xlabel('Change in water level')
plt.ylabel('Water flowing through dam')
plt.grid()

theta = np.array((1,1))
ones = np.ones((1, X_training.shape[0]))
X_training = X_training.transpose()
X_training = np.vstack((ones, X_training))
y_training = y_training.reshape((1, len(y_training)))
lambda_for_regularization = 0

cost = linearregression.compute_cost_with_regularization(theta,
                                                         X_training,
                                                         y_training,
                                                         lambda_regularization= 0)

gradients = linearregression.compute_gradient_with_regularization(theta,
                                                                  X_training,
                                                                  y_training,
                                                                  lambda_regularization=0)

initial_theta = np.array((0,0))

optimized_theta = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta, X_training,
                                                                                    y_training, lambda_for_regularization,
                                                                                    OptimizationAlgo.FMIN_CG)
print(f'Cost at {theta} : {cost}')
print(f'Gradients at {theta} : {gradients}')
print(f'Optimized theta : {optimized_theta}')

prediction = optimized_theta.dot(X_training)

plt.figure(1)
plt.plot(X_training[1], prediction)
plt.show()