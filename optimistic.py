import matplotlib.pyplot as plt
import numpy as np


num_trails = 10000
eps = 0.1
bandit_prob = [0.2,0.5,0.75]

class Bandit:
    def __init__(self, p):
        # p the win rate
        self.p = p
        self.p_estimate = 5
        self.N = 1
        


    def pull(self):
        return np.random.random() < self.p


    def update(self, x):
        self.N +=1
        self.p_estimate = ((self.N -1) * self.p_estimate + x) / self.N



def experiment():

    bandits = [Bandit(p) for p in bandit_prob]

    rewards = np.zeros(num_trails)
    num_optimal = 0

    optimal_j = np.argmax([b.p for b in bandits])

    print("optimal_j:", optimal_j)

    for i in range(num_trails):

      
        j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal +=1


        x = bandits[j].pull()

        rewards[i] = x
        bandits[j].update(x)


    for b in bandits:
        print("mean estimate:", b.p_estimate)

    
    print("total rewards earned:", rewards.sum())
    print("overall wins rate:", rewards.sum()/ num_trails)
    print("num times selected optimal bandit", num_optimal)

    print("num times each bandit selected ", [b.N for b in bandits])


    cummulative_rewards = np.cumsum(rewards)
    wins_rates = cummulative_rewards / (np.arange(num_trails) + 1)
    plt.plot(wins_rates)
    plt.plot(np.ones(num_trails)*np.max(bandit_prob))
    plt.show()


if __name__ == "__main__":
    experiment()



