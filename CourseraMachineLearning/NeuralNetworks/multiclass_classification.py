import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from CourseraMachineLearning.Utility import neuralnetworks

data = scipy.io.loadmat('ex3data1.mat')
print('End')

X = data['X']
y = data['y']

plot_training_data = neuralnetworks.display_data(X, 100)






plt.show()
