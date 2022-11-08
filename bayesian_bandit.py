from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


num_trails = 2000
bandit_prob = [0.2,0.5,0.75]


class Bandit:
    def __init__(self, p):
        # p the win rate
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0
        


    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)


    def update(self, x):
        self.N +=1
        self.a += x
        self.b += 1-x



def plot(bandits, trails):
    x = np.linspace(0,1,200)

    for b in bandits:
        y = beta.pdf(x,b.a, b.b)
        plt.plot(x,y, label=f"real p: {b.p:.4f}, win rate = {b.a -1}/ {b.N}")
    
    plt.title(f"bandit dist. after {trails}")
    plt.legend()
    plt.show()

def experiment():

    bandits = [Bandit(p) for p in bandit_prob]
    sample_points = [5,10,20,50, 100,200,500,1000,1500,1999]
    rewards = np.empty(num_trails)


    optimal_j = np.argmax([b.p for b in bandits])

    print("optimal_j:", optimal_j)


    for i in range(num_trails):

        # thmpson sampling
        j = np.argmax([b.sample() for b in bandits])

        if i in sample_points:
            plot(bandits, i)

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()


        # update rewards 
        rewards[i] = x

        # update the distribution  for tghe bandit whose arm we just pulled 
        bandits[j].update(x)



    print("total rewards earned:", rewards.sum())
    print("overall wins rate:", rewards.sum()/ num_trails)

    print("num times each bandit selected ", [b.N for b in bandits])


    cummulative_rewards = np.cumsum(rewards)
    wins_rates = cummulative_rewards / (np.arange(num_trails) + 1)
    plt.plot(wins_rates)
    plt.plot(np.ones(num_trails)*np.max(bandit_prob))
    plt.show()


if __name__ == "__main__":
    experiment()



