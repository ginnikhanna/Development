from enum import Enum
from collections import namedtuple



class Distribution(Enum):
    bernoulli = 1



distribution_bernoulli_parameters = namedtuple('BernoulliDistribution', ['probability_of_success', 'random_variable_value'])


def get_probability_of_occurence(parameters:distribution_bernoulli_parameters):

    ''' Let us say there is a random variable X which can only take two values, either 0 or 1.
    The probability distribution of such a random variable is modelled using the Bernoulli distribution with probabily p and is written as

    P(X = x,p) = p^x(1-p)^^1-x) for x = 0, 1'''


    pdf = pow(parameters.probability_of_success, parameters.random_variable_value) * pow((1 - parameters.probability_of_success), (1-parameters.random_variable_value))


    return pdf
