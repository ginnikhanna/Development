from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import show
import pandas as pd


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