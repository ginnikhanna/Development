''' A real life example of a binomial distribution
Let's say you have collected the following data from a call center company

Number of calls completed on average by one employee : 50
Probability of success : 0.04
Average revenue to the company per conversion : 20
Total number of employees : 100
Cost/employee : 200
This is actually an example given by Tony Yiu on https://towardsdatascience.com/fun-with-the-binomial-distribution-96a5ecabf65b

We start with the initial analysis of the profit for the call center and move our way to make the profits better by changing the values of the parameters
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


number_of_calls = 50
number_of_employees = 100
success_rate = 0.04
cost_for_employees = 200
revenue_per_call = 100



#Running 1 of these binomial distributions
conversion_from_one_employee = np.random.binomial(number_of_calls, success_rate, number_of_employees)

#Running several of these binomial distributions

conversion_from_all_employees = [np.sum(np.random.binomial(number_of_calls, success_rate, number_of_employees)) for i in range(1000)]

revenue_from_all_employees = [conversion_employee * revenue_per_call - number_of_employees * cost_for_employees
                              for conversion_employee in conversion_from_all_employees]


# Let us try to improve the revenue by increasing the number of calls made
number_of_calls_improved = 60
success_rate = 0.05

conversion_from_all_employees_improved = [np.sum(np.random.binomial(number_of_calls_improved, success_rate, number_of_employees)) for i in range(1000)]

revenue_from_all_employees_improved = [conversion_employee * revenue_per_call - number_of_employees * cost_for_employees
                              for conversion_employee in conversion_from_all_employees_improved]



plt.figure(1)
sns.distplot(revenue_from_all_employees, 20, label = 'Initial Call Center Profits')
sns.distplot(revenue_from_all_employees_improved, 20, label = 'Improved Call Center Profits')
plt.legend()
plt.show()





