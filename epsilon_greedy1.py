import matplotlib.pyplot as plt
import numpy as np


num_trails = 10000
eps = 0.1
bandit_prob = [0.2,0.5,0.75]

class Bandit:
    def __init__(self, p):
        # p the win rate
        self.p = p
        self.p_estimate = 50
        self.N = 0
        


    def pull(self):
        return np.random.random() < self.p


    def update(self, x):
        self.N +=1
        self.p_estimate = ((self.N -1) * self.p_estimate + x) / self.N



def experiment():

    bandits = [Bandit(p) for p in bandit_prob]

    rewards = np.zeros(num_trails)
    nums_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0

    optimal_j = np.argmax([b.p for b in bandits])

    print("optimal_j:", optimal_j)

    for i in range(num_trails):

        if np.random.random() < eps:
            nums_times_explored +=1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited +=1
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
    print("nums_times_explored:",nums_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit", num_optimal)


    cummulative_rewards = np.cumsum(rewards)
    wins_rates = cummulative_rewards / (np.arange(num_trails) + 1)
    plt.plot(wins_rates)
    plt.plot(np.ones(num_trails)*np.max(bandit_prob))
    plt.show()


if __name__ == "__main__":
    experiment()



