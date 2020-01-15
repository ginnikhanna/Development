import csv
import numpy as np
import sys
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

from CourseraMachineLearning.Utility.logisticregression import sigmoid


# LOAD DATA
df = pd.read_csv('ex2data1.csv')
X = np.array((df['Score_Test_1'].to_numpy(), df['Score_Test_2'].to_numpy()))

yes_indices = np.where((df['Admission']) == 1)[0]
no_indices = np.where((df['Admission'])== 0)[0]

df_yes = df.drop(no_indices)
df_no = df.drop(yes_indices)

assert len(df_yes) + len(df_no) == len(df)

y = np.array(df['Admission'])
number_of_samples = len(y)

plt.figure(1)
plt.scatter(df_yes['Score_Test_1'], df_yes['Score_Test_2'], marker = 'o', label = 'Admitted')
plt.scatter(df_no['Score_Test_1'], df_no['Score_Test_2'], marker = 'x', label = 'Not Admitted')
plt.xlabel('Scores Test 1')
plt.ylabel('Score Test 2')
plt.legend()
plt.show()
