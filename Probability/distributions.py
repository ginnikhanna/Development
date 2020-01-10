from enum import Enum
from scipy.special import comb
from collections import namedtuple



class Distribution(Enum):
    bernoulli = 1
    binomial = 2
    geometric = 3
    multinomial = 4

BernoulliDistribution = namedtuple('BernoulliDistribution', ['probability_of_success', 'random_variable_value'])
BinomialDistribution = namedtuple('BinomialDistribution', ['number_of_trials', 'number_of_success_trials', 'probability_of_success'])
GeometricDistribution = namedtuple('GeometricDistribution', ['number_of_failures', 'probability_of_success'])
MultinomialDistribution = namedtuple('MultinomialDistribution', ['total_number_of_outcomes', 'list_of_possible_outcomes', 'probability_of_each_outcome'])

def get_probability_of_occurence(distribution:Distribution, parameters :(BernoulliDistribution, BinomialDistribution, GeometricDistribution, MultinomialDistribution)):

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

    if distribution == Distribution.geometric:
        pdf = pow((1 - parameters.probability_of_success), parameters.number_of_failures - 1) * parameters.probability_of_success
        return pdf

    if distribution == Distribution.multinomial:
        pdf = 1
        for i in range(len(parameters.list_of_possible_outcomes)):
            total_outcomes = sum(parameters.list_of_possible_outcomes)
            pdf *= comb(total_outcomes, parameters.list_of_possible_outcomes[i])
            parameters.list_of_possible_outcomes.pop(0)




def get_mean_and_variance_of_distribution(distribution : Distribution, parameters : (BernoulliDistribution, BinomialDistribution)):

    if distribution == Distribution.bernoulli:
        mean = parameters.probability_of_success
        variance = parameters.probability_of_success(1-parameters.probability_of_success)

        return mean, variance

    if distribution == Distribution.binomial:
        mean = parameters.number_of_trials * parameters.probability_of_success
        variance = parameters.number_of_trials * parameters.probability_of_success * (1 - parameters.probability_of_success)
        return mean, variance

    if distribution == Distribution.geometric:
        mean = 1/ parameters.probability_of_success
        variance = (1 - parameters.probability_of_success)/(pow(parameters.probability_of_success, 2))
        return mean, variance
