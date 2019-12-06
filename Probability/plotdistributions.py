from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show
import pandas as pd
import numpy as np

#from collections import List

def plot_distribution_bernoulli(p_success, p_failure):

    pdf = {
        'Event': ['Success', 'Failure'],
        'P(Event)': [p_success, p_failure],
        'edges_left': [0.0, 0.5],
        'edges_right': [0.5, 1.0]
    }

    df_pdf = pd.DataFrame(pdf, columns=['Event', 'P(Event)', 'edges_left', 'edges_right'])
    df_pdf_src = ColumnDataSource(df_pdf)


    p = figure(plot_width=600,
               plot_height=600,
               title='Probability Distribution',
               x_axis_label='Random Variable Value',
               y_axis_label='Probability Density Function')

    p.quad(source=df_pdf_src,
           bottom=0, top='P(Event)',
           left='edges_left',
           right='edges_right',
           fill_color='red', line_color='black')

    p.xaxis.ticker = [0.25, 0.75]
    p.xaxis.major_label_overrides = {0.25: '0', 0.75: '1'}

    return p



def plot_distribution_binomial(n : int, p_list):

    p = figure(plot_width=600,
               plot_height=600,
               title='Binomial Distribution',
               x_axis_label='Random Variable Value (X)',
               y_axis_label='Probability Density Function P(X = x)')

    left_edges_list = np.arange(0, n, 1)
    right_edges_list = np.arange(1, n + 1, 1)
    n_list = np.arange(n)

    df_pdf = pd.DataFrame(list(zip(n_list, p_list, left_edges_list, right_edges_list, n_list)),
                      columns=['Number of Successeful Trials',
                               'P_Event',
                               'edges_left',
                               'edges_right',
                               'x_ticks'])


    df_pdf_src = ColumnDataSource(df_pdf)
    p.quad(source=df_pdf_src,
           bottom= 0, top='P_Event',
           left='edges_left',
           right='edges_right',
           fill_color='red', line_color='black')

    h = HoverTool(tooltips  = [('P_Event', '@P_Event'),
                               ('edges_left', '@edges_left')])

    p.add_tools(h)
  #  p.xaxis.ticker = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
#    x_ticks = np.arange(0.25, n/2 +  0.25, 0.5)

#    p.axis.ticker = x_ticks
#    p.axis.major_label_overrides = {key : str(value) for (key, value) in zip(x_ticks, np.arange(n))}

    return p

def plot_distribution_geometric(n : int, p_list):

    p = figure(plot_width=600,
               plot_height=600,
               title='Geometric Distribution',
               x_axis_label='Random Variable Value (X)',
               y_axis_label='Probability Density Function P(X = x)')

    left_edges_list = np.arange(0, n, 1)
    right_edges_list = np.arange(1, n + 1, 1)
    n_list = np.arange(n)

    df_pdf = pd.DataFrame(list(zip(n_list, p_list, left_edges_list, right_edges_list, n_list)),
                      columns=['Number of Trials Before Success',
                               'P_Event',
                               'edges_left',
                               'edges_right',
                               'x_ticks'])


    df_pdf_src = ColumnDataSource(df_pdf)
    p.quad(source=df_pdf_src,
           bottom= 0, top='P_Event',
           left='edges_left',
           right='edges_right',
           fill_color='red', line_color='black')

    h = HoverTool(tooltips  = [('P_Event', '@P_Event'),
                               ('edges_left', '@edges_left')])

    p.add_tools(h)
  #  p.xaxis.ticker = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
#    x_ticks = np.arange(0.25, n/2 +  0.25, 0.5)

#    p.axis.ticker = x_ticks
#    p.axis.major_label_overrides = {key : str(value) for (key, value) in zip(x_ticks, np.arange(n))}

    return p

