# %% [markdown]
# In this lab, we will implement Q learning with deep neural nets.

# %%
import numpy as np
import gym
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque
import copy

from tqdm.std import tqdm


env = gym.make('CartPole-v1')
n_state = int(np.prod(env.observation_space.shape))
n_action = env.action_space.n
print("# of state", n_state)
print("# of action", n_action)

# SEED = 1234
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# env.seed(SEED)
# %% [markdown]
# Given certain policy, how can we compute the value function for each state.

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IF_TARGET = True
IF_REPLAY = True
IF_PER = True


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
class Policy():
    def __init__(self, n_state, n_action, eps):
        self.q_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_action)
        )
        self.eps = eps
        self.gamma = 0.95
        if IF_TARGET:
            self.target_net = copy.deepcopy(self.q_net)
            self.target_net.to(device)
            self.target_net.eval()
            self.network_sync_counter = 0
            self.network_sync_freq = 10

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.q_net.to(device)
        if IF_PER:
            self.replaybuff = NaivePrioritizedBuffer(50000)
        else:
            self.replaybuff = ReplayBuffer(50000)

    def update(self, data=None):
        if IF_REPLAY:
            if IF_PER:
                obs, act, reward, next_obs, done, index, weight = self.replaybuff.sample(32)
            else:
                obs, act, reward, next_obs, done = self.replaybuff.sample(32)
        if data:
            obs, act, reward, next_obs, done = data

        if IF_TARGET:
            if(self.network_sync_counter == self.network_sync_freq):
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.network_sync_counter = 0
            self.network_sync_counter += 1
        self.optimizer.zero_grad()
        with torch.no_grad():
            if IF_TARGET:
                y = reward + self.gamma * (1 - done) * \
                    torch.max(self.target_net(next_obs), axis=1)[0]
            else:
                y = reward + self.gamma * (1 - done) * \
                    torch.max(self.q_net(next_obs), axis=1)[0]

        if IF_PER:
            # loss = (y - self.q_net(obs)[range(len(obs)), act]).pow(2) * weight
            loss = (y - self.q_net(obs)[range(len(obs)), act]).pow(2)
            prios = loss + 1e-5
            loss = loss.mean()
            self.replaybuff.update_priorities(index, prios.data.cpu().numpy())
        else:
            loss = F.mse_loss(y, self.q_net(obs)[range(len(obs)), act])
        ####################################
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __call__(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(n_action)

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            Q = self.q_net(state).cpu().numpy()
            act = np.argmax(Q)

        return act

# %%


class ReplayBuffer:
    def __init__(self, size):
        self.buff = deque(maxlen=size)

    def add(self, obs, act, reward, next_obs, done):
        self.buff.append([obs, act, reward, next_obs, done])

    def sample(self, sample_size):
        if(len(self.buff) < sample_size):
            sample_size = len(self.buff)

        sample = random.sample(self.buff, sample_size)
        obs = torch.FloatTensor([exp[0] for exp in sample]).to(device)
        act = torch.LongTensor([exp[1] for exp in sample]).to(device)
        reward = torch.FloatTensor([exp[2] for exp in sample]).to(device)
        next_obs = torch.FloatTensor([exp[3] for exp in sample]).to(device)
        done = torch.FloatTensor([exp[4] for exp in sample]).to(device)
        return obs, act, reward, next_obs, done

    def __len__(self):
        return len(self.buff)

# %%

# Credit to https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb


class NaivePrioritizedBuffer(object):
    def __init__(self, size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = size
        self.pos = 0
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.buff = deque(maxlen=size)

    def add(self, obs, act, reward, next_obs, done):
        max_prio = self.priorities.max() if len(self.buff) > 0 else 1.0
        self.buff.append([obs, act, reward, next_obs, done])
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buff) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buff), batch_size, p=probs)
        sample = [self.buff[idx] for idx in indices]

        total = len(self.buff)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        weights = torch.FloatTensor(weights).to(device)

        # batch = zip(*samples)
        # states = np.concatenate(batch[0])
        # actions = batch[1]
        # rewards = batch[2]
        # next_states = np.concatenate(batch[3])
        obs = torch.FloatTensor([exp[0] for exp in sample]).to(device)
        act = torch.LongTensor([exp[1] for exp in sample]).to(device)
        reward = torch.FloatTensor([exp[2] for exp in sample]).to(device)
        next_obs = torch.FloatTensor([exp[3] for exp in sample]).to(device)
        done = torch.FloatTensor([exp[4] for exp in sample]).to(device)

        return obs, act, reward, next_obs, done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buff)


# %%
losses_list, reward_list = [], []
policy = Policy(n_state, n_action, 1)
update_index = 0
loss = 0
for i in tqdm(range(10000)):
    obs, rew = env.reset(), 0
    while True:
        act = policy(obs)
        next_obs, reward, done, _ = env.step(act)
        rew += reward

        update_index += 1
        if len(policy.replaybuff) > 2e3 and update_index > 4:
            update_index = 0
            loss = policy.update()
        # after using replay buffer
        if IF_REPLAY:
            policy.replaybuff.add(obs, act, reward, next_obs, done)
        #################################################
        else:
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs[None, :]).to(device)
            next_obs = torch.FloatTensor(next_obs[None, :]).to(device)
            done = torch.FloatTensor([done]).to(device)
            loss = policy.update(data=[obs, act, reward, next_obs, done])
        obs = next_obs
        if done:
            break
    if i > 0 and i % 500 == 0:
        print("itr:({:>5d}) loss:{:>3.4f} reward:{:>3.1f}".format(
            i, np.mean(losses_list[-500:]), np.mean(reward_list[-500:])))
    policy.eps = max(0.01, policy.eps - 1.0/20000)

    losses_list.append(loss), reward_list.append(rew)

# %%
policy.eps = 0.0
scores = [sum(run_episode(env, policy, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))

import pandas as pd
df = pd.DataFrame({'loss': losses_list, 'reward': reward_list})
df.to_csv("./ClassMaterials/Lecture_09_QLearning/data/dqn-.csv",
          index=False, header=True)
