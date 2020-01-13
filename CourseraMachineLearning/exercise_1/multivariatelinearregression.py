import csv
import numpy as np
import sys
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pandas as pd
from CourseraMachineLearning import util

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
X, mu, sigma = util.normalized_features_matrix(X)

#ADD THE ROW FOR 1s FOR THETA_0
ones = np.ones((1, number_of_samples))
X = np.vstack((ones, X))

#GET GRADIENT DESCENT
theta = np.zeros((np.size(X, 0), 1))
theta, J = util.gradient_descent_multivariate(X, theta, y, number_of_iterations=400, alpha = 0.1)

#PRICE PREDICTION FOR A 1650 ft2 3 BEDROOM HOUSE
# DO SCALE THE FEATURES BEFORE MAKING A PREDICTION
price_gradient_descent = theta[0] + theta[1]*(1650-mu[0])/sigma[0] + theta[2]*(3 - mu[1])/sigma[1]
print(price_gradient_descent)

#PRICE PREDICTION FOR A 1650 ft2 3 BEDROOM HOUSE USING NORMAL EQUATIONS
X = np.array((df['Size'].to_numpy(), df['Bedrooms'].to_numpy()))
y = np.array(df['Price'])
ones = np.ones((1, number_of_samples))
X = np.vstack((ones, X))

theta_normal_equations = util.parameters_from_normal_equation(X, y)
print(theta_normal_equations)
price_normal_equations = theta_normal_equations[0] + theta_normal_equations[1]*(1650) + theta_normal_equations[2]*3
print(price_normal_equations)