from enum import Enum
from scipy.special import comb
from collections import namedtuple



class Distribution(Enum):
    bernoulli = 1
    binomial = 2



BernoulliDistribution = namedtuple('BernoulliDistribution', ['probability_of_success', 'random_variable_value'])
BinomialDistribution = namedtuple('BinomialDistribution', ['number_of_trials', 'number_of_success_trials', 'probability_of_success'])

def get_probability_of_occurence(distribution:Distribution, parameters :(BernoulliDistribution, BinomialDistribution)):

    ''' Let us say there is a random variable X which can only take two values, either 0 or 1.
    The probability distribution of such a random variable is modelled using the Bernoulli distribution with probabily p and is written as

    P(X = x,p) = p^x(1-p)^^1-x) for x = 0, 1'''

    if distribution == Distribution.bernoulli:
        pdf = pow(parameters.probability_of_success, parameters.random_variable_value) * pow((1 - parameters.probability_of_success), (1-parameters.random_variable_value))

        return pdf

    if distribution == Distribution.binomial:
        pdf = comb(parameters.number_of_trials, parameters.number_of_success_trials) * \
              pow(parameters.probability_of_success, parameters.number_of_success_trials) * \
              pow((1 - parameters.probability_of_success), parameters.number_of_trials - parameters.number_of_success_trials)

        return pdf
