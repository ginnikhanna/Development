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
df = pd.read_csv('ex2data2.csv')


X = np.array((df.Test_1.to_numpy(), df.Test_2.to_numpy()))

yes_indices = np.where((df.Pass) == 1)[0]
no_indices = np.where((df.Pass)== 0)[0]

df_yes = df.drop(no_indices)
df_no = df.drop(yes_indices)
assert len(df_yes) + len(df_no) == len(df)
y = np.array(df.Pass)
number_of_samples = len(y)

# Plot data
plt.figure(1)
plt.scatter(df_yes.Test_1, df_yes.Test_2, marker = 'o', label = 'Passed')
plt.scatter(df_no.Test_1, df_no.Test_2, marker = 'x', label = 'Failed')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

#Looking at the data tells us that the data can not be separated using a line. It requires some other kind of decision boundary.
#One way to fit the data is using more features, and create a polynom with the features. We need to make a base function matrix, which is called mapping the features.
#Let us create this mapfeature matrix, what I would also call the base matrix

X_with_mapped_features = logisticregression.construct_matrix_with_mapped_features(X, degree= 6)
print(X_with_mapped_features.shape)