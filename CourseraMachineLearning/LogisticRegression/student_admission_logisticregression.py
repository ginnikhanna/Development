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

#Load data
df = pd.read_csv('ex2data1.csv')
X = np.array((df['Score_Test_1'].to_numpy(), df['Score_Test_2'].to_numpy()))

yes_indices = np.where((df['Admission']) == 1)[0]
no_indices = np.where((df['Admission'])== 0)[0]

df_yes = df.drop(no_indices)
df_no = df.drop(yes_indices)
assert len(df_yes) + len(df_no) == len(df)
y = np.array(df['Admission'])
number_of_samples = len(y)

# Plot data
plt.figure(1)
plt.scatter(df_yes['Score_Test_1'], df_yes['Score_Test_2'], marker = 'o', label = 'Admitted')
plt.scatter(df_no['Score_Test_1'], df_no['Score_Test_2'], marker = 'x', label = 'Not Admitted')
plt.xlabel('Scores Test 1')
plt.ylabel('Score Test 2')
plt.legend()

# Prepare data to calculate cost function
ones = np.ones((1,number_of_samples))
X = np.vstack((ones, X))

initial_theta = np.zeros(3)
cost = logisticregression.compute_cost(initial_theta, X,  y)
gradients = logisticregression.compute_gradients(initial_theta, X, y)

print(f'Cost at initial theta : {cost}')
print(f'Gradients at initial theta: {gradients}')

# Perform optimization on cost_function to find optimized theta
result = logisticregression.minimize_cost_and_find_theta(initial_theta, X, y)
final_theta = result.x
print (f'Final cost at optimized thetas: {result.fun}')

print(f'Optimized thetas: {result.x}')

# Plot decision boundary
plot_with_decision_boundary = logisticregression.plot_decision_boundary(final_theta, X, y, fig_number = 1)
plt.show()

# Predicting if a student will be admitted or not
# At first we will find the probability of a student with scores 45,85 to be admitted in the program
x = np.array([1, 45, 85])
probability_of_admission = logisticregression.sigmoid(x.dot(final_theta))
print(f'Probability of admission for a student with scores 45 and 85 is : {probability_of_admission}')

prediction_on_training_data_set = logisticregression.predict_outcome_for_given_dataset(final_theta, X)

accuracy = len(np.where((prediction_on_training_data_set == y))[0])/len(y) * 100.0
print(accuracy)