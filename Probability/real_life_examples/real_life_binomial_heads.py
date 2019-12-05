''' Let us say you are given a coin and dont know if it is fair or not.
How would you go about figuring if it is fair or not.
Well, one way is to actually flip the coin 1000 times and calculate the probability of occurence of heads.
If it is equal to number of tails, then it is fairly a fair coin.
On the other hand, if there is a large devitation in number of heads coming up, it can be safe to say that
the coin is manipulated '''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Let us simulate a fair coin

p_list = [0.5, 0.99]
label_list = ['Fair Coin', 'Manipulated Coin']
n = 1000

for i, p in enumerate(p_list):
    heads = [np.sum(np.random.binomial(n, p, 1000)) for i in range(1000)]
    heads = [head/1000 for head in heads]
    plt.figure(1)
    sns.distplot(heads ,
                 bins = 20,
                 label = f'p = {p} and {label_list[i]}')

plt.legend()
plt.show()