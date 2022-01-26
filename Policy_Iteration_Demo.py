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
print("# of actions", env.env.nA)
print("# of states", env.env.nS)
# At state 14 apply action 2

for prob, next_state, reward, done in env.env.P[14][2]:
    print("If apply action 0 under state 14, there is %3.2g probability it will transition to state %d and yield reward as %i" % (
        prob, next_state, reward))

# %% [markdown]
# First, let's write utility functions we will use to run experiments.
# %% [markdown]
# Given certain policy, how can we compute the value function for each state.

# %%


def compute_policy_v(env, policy, gamma=1.0):
    # The goal of this function is to calculate the Q(s,a) for each action under each state
    eps = 1e-10
    V = np.zeros(n_state)
    while True:
        prev_V = np.copy(V)
        for state in range(n_state):
            action = policy[state]
            V[state] = 0
            for prob, next_state, reward, done in env.env.P[state][action]:
                V[state] += prob * (reward + gamma * prev_V[next_state])
        if (np.sum((np.fabs(prev_V - V))) <= eps):
            break
    return V


# %% [markdown]
# Let's test a random policy

# %%
random_policy = [np.random.choice(n_action) for _ in range(n_state)]
print(random_policy)
rand_v = compute_policy_v(env, random_policy)
print(np.round(rand_v, 5))

# %% [markdown]
# Given value function, we need to extract the best policy from it.

# %%


def extract_policy(V, gamma=1.0):
    policy = np.zeros(n_state)
    Q = np.zeros((n_state, n_action))
    for state in range(n_state):
        for action in range(n_action):
            for prob, next_state, reward, done in env.env.P[state][action]:
                Q[state, action] += prob * (reward + gamma*V[next_state])
        policy[state] = np.argmax(Q[state])
    return policy


# %%
print(extract_policy(rand_v))

# %% [markdown]
# Now let's start with a random policy and compute the value then extract new policy. Do this recursively will improve the policy

# %%


def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        # this will calculate the return for the first step
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


# %%
policy = random_policy
max_iterations = 200000
gamma = 1.0
for i in range(max_iterations):
    policy_v = compute_policy_v(env, policy)
    new_policy = extract_policy(policy_v)
    score = run_episode(env, new_policy, gamma, False)
    print("iteration %i: score %f" % (i, score))
    if (np.all(policy == new_policy)):
        print('Policy-Iteration converged at step %d.' % (i+1))
        break
    policy = new_policy

scores = [run_episode(env, policy, gamma, False) for _ in range(100)]
print("Final score:", np.mean(scores))