# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#
# %% [markdown]
# In this lab, we will implement policy iteration and value iteration to solve the Frozen lake problem.
#
# Let's first get ourselves familarized with the environment.

# %%
import numpy as np
import gym


env = gym.make('FrozenLake-v1')  # or you can try 'FrozenLake8x8-v0'
env.render()
n_state = env.env.nS
n_action = env.env.nA

P = env.env.P
Q = np.zeros((n_state, n_action))
V = np.zeros(n_state)
gamma = 1
eps = 1e-10
i = 0

def updateValue(V, gamma) :
    newV = np.empty_like(V)
    for state in range(n_state) :
        for action in range(n_action) :
            acc = 0
            for prob, next_state, reward, done in env.env.P[state][action]:
                   acc += prob * (reward + (V[next_state] * gamma))
            Q[state, action] = acc
        newV[state] = max(Q[state]) 

    return newV

while True:
    i = i + 1
    oldV = np.copy(V)
    V = updateValue(V, gamma)
    if (np.sum((np.fabs(oldV - V))) <= eps):
            break
print(V)
print(i)

assert all(abs(v - 0.82352941) < 1e-5 for v in V[0:5]), "Value function is not calculated correctly"
assert all(abs(v - 0) < 1e-5 for v in V[[5,7,11,12,15]]), "Value function is not calculated correctly"
