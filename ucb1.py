import matplotlib.pyplot as plt
import numpy as np


num_trails = 10000
eps = 0.1
bandit_prob = [0.2,0.5,0.75]

class Bandit:
    def __init__(self, p):
        # p the win rate
        self.p = p
        self.p_estimate = 0
        self.N = 0
        


    def pull(self):
        return np.random.random() < self.p


    def update(self, x):
        self.N +=1
        self.p_estimate = ((self.N -1) * self.p_estimate + x) / self.N



def ucb(mean, n, nj):
    return mean + np.sqrt(2* np.log(n)/ nj)



def experiment():

    bandits = [Bandit(p) for p in bandit_prob]
    rewards = np.empty(num_trails)
    total_plays = 0

    optimal_j = np.argmax([b.p for b in bandits])

    print("optimal_j:", optimal_j)


    # initialization : play each bandit once 
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)


    for i in range(num_trails):

        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        rewards[i] = x
        bandits[j].update(x)


    for b in bandits:
        print("mean estimate:", b.p_estimate)

    
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



