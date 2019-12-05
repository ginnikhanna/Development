The repository consists of code for basic probability ditributions


1. Bernoulli Distribution

Such a distribution is useful for modelling outcomes of processes which ask the question of Success of Failure.
The process has only two outcomes. This can be then used for answering the following questions :

A. Is the kid going to be a boy or a girl
B. Does the man have cancer or not
C. Is this a cat or a dog ?
D. Is the transmitted bit 0 or 1

The Bernoulli distribution needs only one paramter, called the probability of success, which is denoted by 'p'.

The probability mass function for the Bernoulli distribution is given as :

P(X = x) = p^x(1-p)^(1-x) for x = {0,1}
E[X] = p
Var[X] = p(1-p)

Bernoulli distribution is the simplest distribution which forms the basis for a lot of other distributions, the next one which we
will study is the Binomial Distribution.


AIM : Use the Bernoulli distribution in situation where only one trial is concerned and the outcome is either 1 or 0.

2. Binomial Distribution
The Binomial Distribution is actually repetition of several independent events with a Bernoulli distribution. This distribution has its significance in
assessing the probabilites of repeated independent trials. The PDF of the process modelled with binomial distributin can gives us the abilitiy to assess the
probability of a range of potential results. It has specific application in risk management.

It has the potential to answer questions such as :

A. Is the coin fair or not ? In this case, actually the coin can be flipped multiple times and the probability of occurence of heads on each flip can be recorded. If the outcome is anything other than
close to 0 for a coin flipped 1000 times, it can be safely concluded that the coin is unfair.

B. Is the team going to win 5 games in a row ?

C. Can the couple get 5 boys in a row ?

D. In a manufacturing firm, what is the probability that all the 100 manufactured bulbs are defective.

It has two requirements:

1. There are exactly two outcomes, success or failure
2. The probability of success is same in all the trials.

For a Binomial distribution with 'successes' from 'n' Bernoulli experiments, the probability that 'k' trials result is success is:
P(X = k) = nCk * (p^k)(1-p)^(n-k)

E(X) = n*p
Var(X) = np(1-p)

AIM : Use Binomial distribution in situations where probability of success is known and the question to be answered is "How many of those n trials will result in a success".
It is also useful to test for statistical significance by finding out any discrepancies in the underlying distrubution of the physical experiment.
Actually if you make 'n' very high, the binomial distribution approaches the normal distribution.


3. Geometric Distribution
git