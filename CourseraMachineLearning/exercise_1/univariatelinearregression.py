import csv
import pandas as pd
from CourseraMachineLearning import util
import matplotlib.pyplot as plt
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

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


show(p)
