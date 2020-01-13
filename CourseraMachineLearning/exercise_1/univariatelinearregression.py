import csv
import numpy as np
import sys

import pandas as pd
from CourseraMachineLearning import util
import matplotlib.pyplot as plt
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

FLT_EPSILON = 1e-4

# Load data
df = pd.read_csv('ex1data1.csv')

# Plot data
df_src = ColumnDataSource(df)

p = figure(plot_width=600,
           plot_height=600,
           title=' Profit vs Populations',
           x_axis_label='Population',
           y_axis_label='Profit')

p.circle(x = 'Population',
         y = 'Profit',
         source = df_src)
#show(p)

#COMPUTE COST FUNCTION
#Initial settings
X = df['Population'].to_numpy()
ones = np.ones_like(X)

#Making column of ones for theta_0
X = np.vstack((np.ones_like(X), X))

theta =  np.zeros((2,1))
y = df['Profit'].to_numpy()
number_of_training_samples = len(df['Population'])

#TODO: put a unit test here
J_theta = np.mean((theta.transpose().dot(X) - y)**2)/(2)
assert J_theta - 32.0727 < FLT_EPSILON
