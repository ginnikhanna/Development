from __future__ import division
import csv
import numpy as np
import sys
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

from CourseraMachineLearning.Utility.logisticregression import sigmoid
from CourseraMachineLearning.Utility import logisticregression

# Load data
df = pd.read_csv('ex2data2.csv')

X = np.array((df.Test_1.to_numpy(), df.Test_2.to_numpy()))

yes_indices = np.where((df.Pass) == 1)[0]
no_indices = np.where((df.Pass) == 0)[0]

df_yes = df.drop(no_indices)
df_no = df.drop(yes_indices)
assert len(df_yes) + len(df_no) == len(df)
y = np.array(df.Pass)
number_of_samples = len(y)

# Plot data
plt.figure(100)
plt.scatter(df_yes.Test_1, df_yes.Test_2, marker='o', label='Passed')
plt.scatter(df_no.Test_1, df_no.Test_2, marker='x', label='Failed')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()

# Looking at the data tells us that the data can not be separated using a line. It requires some other kind of decision boundary.
# One way to fit the data is using more features, and create a polynom with the features.
# We need to make a base function matrix, which is called mapping the features.
# Let us create this mapfeature matrix, what I would also call the base matrix

X_with_mapped_features = logisticregression.construct_matrix_with_mapped_features(X, degree=6)

initial_theta = np.zeros(X_with_mapped_features.shape[0])
cost = logisticregression.compute_cost_with_regularization(initial_theta, X_with_mapped_features, y,
                                                           lambda_for_regularization=1)
gradients = logisticregression.compute_gradients_with_regularization(initial_theta, X_with_mapped_features, y,
                                                                     lambda_for_regularization=1)
print(f'Cost at initial theta : {cost}')
print(f'Gradients at initial theta: {gradients}')

# Perform optimization on cost_function to find optimized theta
result = logisticregression.minimize_cost_and_find_theta_with_regularization(initial_theta,
                                                                             X_with_mapped_features,
                                                                             y, lambda_for_regularization=1)
final_theta = result.x
print(f'Final cost at optimized thetas: {result.fun}')
print(f'Optimized thetas: {result.x}')

# Plot decision boundary
plot_with_decision_boundary = logisticregression.plot_decision_boundary_contours(final_theta, X, y, 1, color='k')

# Calculate accuracy on predicted results
prediction_on_training_data_set = logisticregression.predict_outcome_for_given_dataset(final_theta,
                                                                                       X_with_mapped_features)
accuracy = logisticregression.get_accuracy(prediction_on_training_data_set, y)
print(f'Accuracy: {accuracy}')

# Running optimization routine for different values of theta
lambda_for_regularization_list = [0, 1, 10, 100]
labels = [str(lambda_for_regularization_list_i) for lambda_for_regularization_list_i in lambda_for_regularization_list]
colors = ['r', 'g', 'b', 'k']
h = []

for index, lambda_i in enumerate(lambda_for_regularization_list):
    # Plot data
    plt.figure(index)
    plt.scatter(df_yes.Test_1, df_yes.Test_2, marker='o', label='Passed')
    plt.scatter(df_no.Test_1, df_no.Test_2, marker='x', label='Failed')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()

    initial_theta = np.zeros(X_with_mapped_features.shape[0])
    cost = logisticregression.compute_cost_with_regularization(initial_theta, X_with_mapped_features, y, lambda_i)
    gradients = logisticregression.compute_gradients_with_regularization(initial_theta, X_with_mapped_features, y,
                                                                         lambda_i)
    result = logisticregression.minimize_cost_and_find_theta_with_regularization(initial_theta, X_with_mapped_features,
                                                                                 y, lambda_i)
    final_theta = result.x
    prediction_on_training_data_set = logisticregression.predict_outcome_for_given_dataset(final_theta,
                                                                                           X_with_mapped_features)
    accuracy = logisticregression.get_accuracy(prediction_on_training_data_set, y)
    print(f'Accuracy for {lambda_i} is {accuracy}')

    plot_with_decision_boundary = logisticregression.plot_decision_boundary_contours(final_theta, X, y, index, colors[index])
    plt.title(f'Decision boundary with {lambda_i}')

plt.show()
