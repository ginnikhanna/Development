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

# LOAD DATA
df = pd.read_csv('ex1data1.csv')

# PLOT DATA
df_src = ColumnDataSource(df)

p = figure(plot_width=600,
           plot_height=600,
           title=' Profit vs Populations',
           x_axis_label='Population in 10000s',
           y_axis_label='Profit in 10000s $ ',
        )

p.circle(x = 'Population',
         y = 'Profit',
         source = df_src,
         legend_label='Data Points'
         )
#show(p)

# COMPUTE COST FUNCTION
X = df['Population'].to_numpy()
y = df['Profit'].to_numpy()
ones = np.ones_like(X)
X = np.vstack((np.ones_like(X), X))
theta =  np.zeros((2,1))

#TODO: put a unit test here
J_theta = util.compute_univariate_cost_function(X, theta, y)
assert J_theta - 32.0727 < FLT_EPSILON

theta = np.array((-1,2))
J_theta = util.compute_univariate_cost_function(X, theta, y)
assert J_theta - 54.2425 < FLT_EPSILON

# GRADIENT DESCENT
theta = np.zeros((2,1))
number_of_iterations = 1500
alpha = 0.01

theta, J = util.gradient_descent_univariate(X, theta, y, number_of_iterations, alpha)
print(f'Final theta {theta} with cost function {J}')

#PLOT FITTED DATA TO THE LINE
predicted_profit = theta.transpose().dot(X)
df['Predicted_Profit'] = pd.Series(predicted_profit[0], index=df.index)

df_src = ColumnDataSource(df)
p.line(x = 'Population',
        y = 'Predicted_Profit',
       color = 'red',
       source = df_src,
       legend_label ='Predicted Model'
       )
show(p)

#PREDICT VALUES FOR POPULATION OF 35,000 and 70,000
profit_1 = np.array([1, 3.5]).transpose().dot(theta) * 10000
profit_2 = np.array([1, 7.0]).transpose().dot(theta) * 10000

print(f'Profit for population of 35000 is {profit_1}')
print(f'Profit for population of 70000 is {profit_2}')