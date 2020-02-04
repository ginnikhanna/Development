import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io

import pandas as pd
from CourseraMachineLearning.Utility import linearregression, diagnostics
from CourseraMachineLearning.Utility.linearregression import OptimizationAlgo


#Load Data
data = scipy.io.loadmat('ex5data1.mat')

X_training_raw = data['X']
X_val_raw = data['Xval']
X_test = data['Xtest']

y_training = data['y']
y_val = data['yval']
y_test = data['ytest']

# Plot Data

plt.figure(1)
plt.scatter(X_training_raw, y_training)
plt.xlabel('Change in water level')
plt.ylabel('Water flowing through dam')
plt.grid()

theta = np.array((1,1))
ones = np.ones((1, X_training_raw.shape[0]))
X_training = X_training_raw.transpose()
X_training = np.vstack((ones, X_training))
y_training = y_training.reshape((1, len(y_training)))

ones = np.ones((1, X_val_raw.shape[0]))
X_val = X_val_raw.transpose()
X_val = np.vstack((ones, X_val))
y_val = y_val.reshape((1, len(y_val)))

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


error_training, error_cross_val = diagnostics.get_training_error(X_training,
                                                                 y_training,
                                                                 X_val,
                                                                 y_val)

plt.figure(2)
plt.plot(error_training, label = 'Training error')
plt.plot(error_cross_val, label = 'Cross validation error')
plt.grid()


lambda_for_regularization = 0

X_poly_training = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X_training_raw.transpose(), 8)
X_poly_training = linearregression.normalized_features_matrix(X_poly_training)
ones = np.ones((1, X_poly_training.shape[1]))
X_poly_training = np.vstack((ones, X_poly_training))

X_poly_val = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X_val_raw.transpose(), 8)
X_poly_val = linearregression.normalized_features_matrix(X_poly_val)
ones = np.ones((1, X_poly_val.shape[1]))
X_poly_val = np.vstack((ones, X_poly_val))

initial_theta = np.zeros((X_poly_training.shape[0]))

optimized_theta = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                                    X_poly_training,
                                                                                    y_training,
                                                                                    lambda_for_regularization,
                                                                                    OptimizationAlgo.FMIN_CG)

prediction_poly = optimized_theta.dot(X_poly_training)

plt.figure(1)
plt.scatter(X_training[1], prediction_poly)


error_training_poly, error_cross_val_poly = diagnostics.get_training_error(X_poly_training,
                                                                 y_training,
                                                                 X_poly_val,
                                                                 y_val)

plt.figure(2)
plt.plot(error_training_poly, label = 'Training error with polynomial regression')
plt.plot(error_cross_val_poly, label = 'Cross validation error with polynomial regression')
plt.grid()
plt.legend()
plt.show()

