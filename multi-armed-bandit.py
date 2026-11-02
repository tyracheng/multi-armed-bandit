"""
Adapted from multiarmed_bandit.py by Anson Wong (ankonzoid)
https://github.com/ankonzoid/LearningX/blob/master/classical_RL/multiarmed_bandit/multiarmed_bandit.py

Modifications: 
- Made it so that arm win rates change randomly after each pull to simulate a changing environment 
- Added an agent with a fixed step size to simulate an agent adapting to the changing environment 
- Compared both agents in a steady, unchanging world versus a changing world 
"""

import os
import numpy as np
import matplotlib.pyplot as plt 

probs = [0.10, 0.50, 0.60, 0.80, 0.10,
         0.25, 0.60, 0.45, 0.75, 0.65] # adjust and play around with later
N_steps = 500
N_experiments = 1000
eps = 0.1
alpha = 0.3
save_fig = True # save in same directory
output_dir = os.path.join(os.getcwd(), "output")

rng = np.random.default_rng(seed = 0)

class Environment:
    def __init__(self, probs, change = 0.0): # change is the amount of change in environment 
        self.probs = np.array(probs, dtype = float)
        self.change = change # how fast the arms change (0 means its probability win rate never changes)
    
    def steps(self, action):
        reward = 1 if (rng.random() < self.probs[action]) else 0
        # if the change is > 0 (if theres change), change probability of arms 
        if self.change > 0:
            self.probs += rng.normal(0.0, self.change, size = self.probs.shape) # self.change affects standard deviation, 
            # so this showcases how the change affects the environment because it controls how much the environment varies
            # Without size, you’d get a single number, and adding it to the whole array would nudge every arm by the same 
            # amount at the same time. All arms would rise and fall together, and the best arm would never change.
            self.probs = np.clip(self.probs, 0.05, 0.95) # wander between 5% and 95%
        return reward


class Agent:
    def __init__(self, nActions, eps, alpha = None):
        self.nActions = nActions # number of actions
        self.eps = eps # epsilon in the e-greedy strategy
        self.alpha = alpha
        self.n = np.zeros(nActions, dtype = int) # action counts n(a)
        self.Q = np.zeros(nActions, dtype = float) # value Q(a). prediction of probability

    def update_Q(self, action, reward):
        self.n[action] += 1
        step = (1.0/self.n[action]) if self.alpha is None else self.alpha
        self.Q[action] += step * (reward - self.Q[action])

    def get_action(self):
        # epsilon-greedy 
        if rng.random() < self.eps:
            return rng.integers(self.nActions)
        else:
            return rng.choice(np.flatnonzero(self.Q == self.Q.max()))


def experiment(probs, N_episodes, make_agent, change = 0.0): # an episode is a single full run through an experiment 
    # start multi-armed bandit 
    env = Environment(probs, change)
    agent = make_agent(len(probs))
    actions, rewards = [], []
    for episodes in range(N_episodes):
        action = agent.get_action()
        reward = env.steps(action)
        agent.update_Q(action, reward) 
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)

def make_average_agent(n):
    return Agent(n, eps)

def make_fixed_step_agent(n):
    return Agent(n, eps, alpha = alpha)

agents = {
    "Average (1/n)":    make_average_agent,
    f"Fixed step ({alpha})": make_fixed_step_agent,
}
worlds = {
    "Steady world (change = 0)":      0.0,
    "Changing world (change = 0.01)": 0.03,
}

def average_rewards(make_agent, change):
    R = np.zeros(N_steps)
    for i in range(N_experiments):
        _, rewards = experiment(probs, N_steps, make_agent, change)
        R += rewards
    return R/N_experiments

    
steady_average = average_rewards(make_average_agent, 0.0)
steady_fixed = average_rewards(make_fixed_step_agent, 0.0)

changing_average = average_rewards(make_average_agent, 0.03)
changing_fixed = average_rewards(make_fixed_step_agent, 0.03)

os.makedirs(output_dir, exist_ok = True)

plt.plot(steady_average, label = "Average (1/n)")
plt.plot(steady_fixed, label = f"Fixed step ({alpha})")
plt.title("Steady World")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "steady_world.png"), bbox_inches = "tight")
plt.close()

plt.plot(changing_average, label = "Average (1/n)")
plt.plot(changing_fixed, label = f"Fixed step ({alpha})")
plt.title("Changing World")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "changing_world.png"), bbox_inches = "tight")
plt.close()


print("Sensitivity to change size: when a constant overtakes 1/n ")
for change in [0.0, 0.01, 0.03, 0.05, 0.1]:
    avg = average_rewards(make_average_agent, change)
    fixed = average_rewards(make_fixed_step_agent, change)
    print(change, ": ", avg[-100:].mean(), fixed[-100:].mean())

print("Sensitivity to step size: finding the best balance for the change level")
for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
    fixed = average_rewards(make_fixed_step_agent, 0.05)
    print(alpha, ": ", fixed[-100:].mean())

