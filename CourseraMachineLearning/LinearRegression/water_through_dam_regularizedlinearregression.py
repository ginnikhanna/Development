import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io

import pandas as pd
from CourseraMachineLearning.Utility import linearregression

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
plt.show()

theta = np.ones((2,1))
ones = np.ones((1, X_training.shape[0]))
X_training = X_training.transpose()
X_training = np.vstack((ones, X_training))
y_training = y_training.reshape((1, len(y_training)))

cost = linearregression.compute_cost_with_regularization(X_training,
                                                         theta,
                                                         y_training,
                                                         lambda_regularization= 0)
print(f'Cost at {theta} : {cost}')