# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# In this lab, we will implement MC policy iteration. Essentially, what we need to do is run policy iteration without `env.env.P`.

# %%
import numpy as np
from tqdm import tqdm
import gym

env = gym.make('FrozenLake-v1')  # or you can try 'FrozenLake8x8-v0'
env.render()
n_state = env.env.nS
n_action = env.env.nA
print("# of actions", env.env.nA)
print("# of states", env.env.nS)

# %% [markdown]
# Given certain policy, how can we compute the value function for each state.

# %%


def run_episode(env, policy, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    states = []
    rewards = []
    actions = []
    while True:
        if render:
            env.render()
        states.append(obs)
        action = int(policy(obs))
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    return states, actions, rewards


# %%
class Policy:
    def __init__(self, Q, N, eps):
        self.Q = Q
        self.N = N
        self.eps = eps
        self.gamma = 0.98

    def update(self, states, actions, rewards):
        returns = np.zeros_like(rewards)
        s = 0
        for i in reversed(range(len(returns))):
            s = s * self.gamma + rewards[i]
            returns[i] = s

        for state, action, rs in zip(states, actions, returns):
            self.N[state, action] += 1
            self.Q[state, action] = (
                self.Q[state, action] * (self.N[state, action] - 1) + rs)/self.N[state, action]

    def __call__(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(n_action)

        return np.argmax(self.Q[state, :])

# %% [markdown]
# Let's start to train the Q table.


# %%
Q = np.zeros((n_state, n_action))
N = np.zeros_like(Q)
policy = Policy(Q, N, 1)
for i in tqdm(range(20000)):
    states, actions, rewards = run_episode(env, policy)
    policy.update(states, actions, rewards)
    policy.eps = max(0.01, policy.eps - 1.0/20000)

policy.eps = 0.0
scores = [sum(run_episode(env, policy, False)[2]) for _ in range(100)]
print("Final score: {:.2f}".format(np.mean(scores)))
