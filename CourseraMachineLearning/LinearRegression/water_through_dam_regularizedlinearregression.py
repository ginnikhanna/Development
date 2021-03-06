import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import interp1d

import pandas as pd
from CourseraMachineLearning.Utility import linearregression, diagnostics
from CourseraMachineLearning.Utility.linearregression import OptimizationAlgo


#Load Data
data = scipy.io.loadmat('ex5data1.mat')

X_training_raw = data['X']
X_val_raw = data['Xval']
X_test_raw = data['Xtest']

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
y_training = y_training.flatten()

ones = np.ones((1, X_val_raw.shape[0]))
X_val = X_val_raw.transpose()
X_val = np.vstack((ones, X_val))
y_val = y_val.flatten()

y_test = y_test.flatten()


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
                                                                 y_val,
                                                                 lambda_for_regularization)

plt.figure(2)
plt.plot(error_training, label = 'Training error')
plt.plot(error_cross_val, label = 'Cross validation error')
plt.grid()


lambda_for_regularization = 0

X_poly_training = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X_training_raw.transpose(), 8)
X_poly_training, mu, sigma = linearregression.normalized_features_matrix(X_poly_training)
ones = np.ones((1, X_poly_training.shape[1]))
X_poly_training = np.vstack((ones, X_poly_training))

X_poly_val = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X_val_raw.transpose(), 8)
X_poly_val = (X_poly_val.transpose() - mu)/sigma
X_poly_val = X_poly_val.transpose()
ones = np.ones((1, X_poly_val.shape[1]))
X_poly_val = np.vstack((ones, X_poly_val))

X_poly_test = linearregression.get_polynomial_feature_matrix_for_univariate_feature_matrix(X_test_raw.transpose(), 8)
X_poly_test = (X_poly_test.transpose() - mu)/sigma
X_poly_test = X_poly_test.transpose()
ones = np.ones((1, X_poly_val.shape[1]))
X_poly_test = np.vstack((ones, X_poly_test))

initial_theta = np.zeros((X_poly_training.shape[0]))


plt.figure(3)
X_training_plot = np.linspace(X_training[1].min(), X_training[1].max(), 100)
f = interp1d(X_training[1], y_training, kind='quadratic')
y_training_plot = f(X_training_plot)
plt.plot(X_training_plot, y_training_plot, label='Training Data', marker='x', linestyle='--')

lambda_for_regularization = [0, 1, 3]

for lambda_for_regularization_i in lambda_for_regularization:

    optimized_theta = linearregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                                        X_poly_training,
                                                                                        y_training,
                                                                                        lambda_for_regularization_i,
                                                                                        OptimizationAlgo.FMIN_CG)

    prediction_poly = optimized_theta.dot(X_poly_training)

    plt.figure(3)
    plt.scatter(X_training[1], prediction_poly, label = f'Poly fit with lambda = {lambda_for_regularization_i}', linestyle = '--')
    plt.grid()
    plt.legend()


    error_training_poly, error_cross_val_poly = diagnostics.get_training_error(X_poly_training,
                                                                     y_training,
                                                                     X_poly_val,
                                                                     y_val,
                                                                     lambda_for_regularization_i)

    plt.figure(4)
    plt.plot(error_training_poly, label = f'Training error with polynomial regression with lambda = {lambda_for_regularization_i}', linestyle = '--')
    plt.plot(error_cross_val_poly, label = f'Cross validation error with polynomial regression = {lambda_for_regularization_i}')
    plt.grid()
    plt.legend()

    test_error = np.mean((optimized_theta.dot(X_poly_test) - y_test)**2)/2
    print(f'lambda = {lambda_for_regularization_i}, test error = {test_error}')



lambda_for_regularization = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


error_training_poly, error_cross_val_poly = diagnostics.get_validation_error(X_poly_training,
                                                                             y_training,
                                                                             X_poly_val,
                                                                             y_val,
                                                                             lambda_for_regularization)


plt.figure(5)
plt.plot(lambda_for_regularization, error_training_poly, label = 'Validation error in training')
plt.plot(lambda_for_regularization, error_cross_val_poly, label = 'Validation error in cross validation')

plt.show()



