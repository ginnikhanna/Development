import csv
import numpy as np
import sys
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pandas as pd
from CourseraMachineLearning.Utility import linearregression

FLT_EPSILON = 1e-4

df = pd.read_csv('ex1data2.csv')
X = np.array((df['Size'].to_numpy(), df['Bedrooms'].to_numpy()))
y = np.array(df['Price'])
number_of_samples = len(y)

# Plot the features
plt.figure(1)
plt.subplot(2,1,1)
plt.scatter(df['Size'], df['Price'])
plt.xlabel('Size of house (ft2)')
plt.ylabel('Price in $s')

plt.subplot(2,1,2)
plt.scatter(df['Bedrooms'], df['Price'])
plt.xlabel('Number of bedrooms')
plt.ylabel('Price in $s')


# NORMALIZE FEATURES
X_norm, mu, sigma = linearregression.normalized_features_matrix(X)

#ADD THE ROW FOR 1s FOR THETA_0
ones = np.ones((1, number_of_samples))
X_norm = np.vstack((ones, X_norm))

#GET GRADIENT DESCENT
theta = np.zeros((np.size(X_norm, 0), 1))
theta, J, _ = linearregression.gradient_descent_multivariate(X_norm, theta, y, number_of_iterations=400, alpha = 0.1)

#PRICE PREDICTION FOR A 1650 ft2 3 BEDROOM HOUSE
# DO SCALE THE FEATURES BEFORE MAKING A PREDICTION
price_gradient_descent = theta[0] + theta[1]*(1650-mu[0])/sigma[0] + theta[2]*(3 - mu[1])/sigma[1]
print(price_gradient_descent)

#PRICE PREDICTION FOR A 1650 ft2 3 BEDROOM HOUSE USING NORMAL EQUATIONS
X = np.array((df['Size'].to_numpy(), df['Bedrooms'].to_numpy()))
y = np.array(df['Price'])
ones = np.ones((1, number_of_samples))
X = np.vstack((ones, X))

theta_normal_equations = linearregression.parameters_from_normal_equation(X, y)
print(theta_normal_equations)
price_normal_equations = theta_normal_equations[0] + theta_normal_equations[1]*(1650) + theta_normal_equations[2]*3
print(price_normal_equations)

# CALCULATE COST FUNCTION vs LEARNING RATE
alpha_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
number_of_iterations = 100

for alpha in alpha_list:
    theta = np.zeros((np.size(X_norm, 0), 1))
    theta, J, J_list = linearregression.gradient_descent_multivariate(X_norm, theta, y, number_of_iterations, alpha)
    plt.figure(5)
    plt.plot(J_list, label = f'alpha = {alpha}')

plt.legend()
plt.ylabel('Cost function J')
plt.xlabel('Number of iterations')
plt.legend(loc = 'best')
plt.grid()
plt.show()
