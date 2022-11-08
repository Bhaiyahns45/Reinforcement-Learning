from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm


num_trails = 2000
bandit_mean= [1,2,3]
# bandit_mean= [5,10,20]


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.predicted_mean = 0 # m in theory lecture ( mean of the mean of the X)
        self.lambda_ = 1
        self.sum_x = 0 # for convenience
        self.tau = 1 
        self.N = 0 
        


    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean


    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.predicted_mean = self.tau * self.sum_x / self.lambda_
        self.N +=1



def plot(bandits, trails):
    x = np.linspace(-3,6,200)

    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1. / b.lambda_))
        plt.plot(x,y, label=f"real mean: {b.true_mean:.4f}, nums plays = {b.N}")
    
    plt.title(f"bandit dist. after {trails}")
    plt.legend()
    plt.show()

def experiment():

    bandits = [Bandit(m) for m in bandit_mean]
    sample_points = [5,10,20,50, 100,200,500,1000,1500,1999]
    rewards = np.empty(num_trails)


    optimal_j = np.argmax([b.true_mean for b in bandits])

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
    plt.plot(np.ones(num_trails)*np.max(bandit_mean))
    plt.show()


if __name__ == "__main__":
    experiment()



