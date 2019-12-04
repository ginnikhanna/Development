import distributions
import plotdistributions
from bokeh.io import show, output_notebook




p = 0.7


bernoulli_parameters_success = distributions.distribution_bernoulli_parameters(p, 1)
bernoulli_parameters_failure = distributions.distribution_bernoulli_parameters(p, 0)


p_success = distributions.get_probability_of_occurence(bernoulli_parameters_success)
p_failure = distributions.get_probability_of_occurence(bernoulli_parameters_failure)


# Arrange data in pandas data fram


plot_bernoulli = plotdistributions.plot_distribution_bernoulli(p_success, p_failure)


