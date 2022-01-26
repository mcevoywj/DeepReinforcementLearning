# %% [markdown]

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
from torch.distributions.normal import Normal


env = gym.make('Pendulum-v0')
n_state = int(np.prod(env.observation_space.shape))
n_action = int(np.prod(env.action_space.shape))
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
        action = policy(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    return states, actions, rewards


# %%
class Policy():
    def __init__(self, n_state, n_action):
        self.q_net1 = nn.Sequential(
            nn.Linear(n_state + n_action, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.q_net2 = nn.Sequential(
            nn.Linear(n_state + n_action, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.actNet = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_action * 2),
        )
        self.q_net1_target = copy.deepcopy(self.q_net1)
        self.q_net2_target = copy.deepcopy(self.q_net2)
        # self.noise = 2
        self.noise = 1.75
        self.act_lim = 2
        self.alpha = 0
        self.gamma = 0.95
        self.ro = 0.9

        self.q_net1_optimizer = torch.optim.Adam(self.q_net1.parameters(), lr=1e-3)
        self.q_net2_optimizer = torch.optim.Adam(self.q_net2.parameters(), lr=1e-3)
        self.actNet_optimizer = torch.optim.Adam(self.actNet.parameters(), lr=1e-3)

        self.replaybuff = ReplayBuffer(50000)

        self.q_net1.to(device)
        self.q_net1_target.to(device)
        self.q_net2.to(device)
        self.q_net2_target.to(device)
        self.actNet.to(device)

        

    def update(self):
        
        obs, act, reward, next_obs, done = self.replaybuff.sample(64)
        output = self.actNet(next_obs)
        mu_ = self.act_lim * torch.tanh(output[:, :n_action])
        var_ = torch.abs(output[:, n_action:]).clamp(1e-3, 2)
        dist_ = Normal(mu_, var_)
        act_ = dist_.sample()
        logprob_ = dist_.log_prob(act_).squeeze()
        with torch.no_grad():
            q_input = torch.cat([next_obs, act_], axis=1)
            y1 = reward + self.gamma * (1 - done) * self.q_net1_target(q_input).squeeze()
            y2 = reward + self.gamma * (1 - done) * self.q_net2_target(q_input).squeeze()
            y = torch.min(y1, y2) - self.alpha*logprob_
        
        q_input = torch.cat([obs, act], axis=1)

        self.q_net1_optimizer.zero_grad()
        Q_net1 = self.q_net1(q_input).squeeze()
        q_net1_loss = F.mse_loss(Q_net1, y)
        q_net1_loss.backward()
        self.q_net1_optimizer.step()

        self.q_net2_optimizer.zero_grad()
        Q_net2 = self.q_net2(q_input).squeeze()
        q_net2_loss = F.mse_loss(Q_net2, y)
        q_net2_loss.backward()
        self.q_net2_optimizer.step()
        
        self.actNet_optimizer.zero_grad()
        output = self.actNet(obs)
        mu = self.act_lim * torch.tanh(output[:, :n_action])
        var = torch.abs(output[:, n_action:]).clamp(1e-3, 2)
        dist = Normal(mu, var)
        act = dist.rsample()
        q_input = torch.cat([obs, act], axis=1)
        min_q = torch.min(self.q_net1(q_input), self.q_net2(q_input))
        actNet_loss = (self.alpha * logprob_ - min_q).mean()
        actNet_loss.backward()
        self.actNet_optimizer.step()

        self._copy_nets()
        return q_net1_loss.item(), q_net2_loss.item(), actNet_loss.item()
        
    def _copy_nets(self):
        for target_param, real_param in zip(self.q_net1_target.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.ro * target_param.data + ((1 - self.ro) * real_param.data))
        for target_param, real_param in zip(self.q_net2_target.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.ro * target_param.data + ((1 - self.ro) * real_param.data))

    def __call__(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # calculate old logprob
            output = self.actNet(state)
            mu = self.act_lim*torch.tanh(output[:n_action])
            var = torch.abs(output[n_action:])
            dist = Normal(mu, var)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim)


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
        act = torch.FloatTensor([exp[1] for exp in sample]).to(device)
        reward = torch.FloatTensor([exp[2] for exp in sample]).to(device)
        next_obs = torch.FloatTensor([exp[3] for exp in sample]).to(device)
        done = torch.FloatTensor([exp[4] for exp in sample]).to(device)
        return obs, act, reward, next_obs, done

    def __len__(self):
        return len(self.buff)

# %%
loss_q_net1_list,loss_q_net2_list , loss_actNet_list, reward_list = [], [], [], []
update_index = 0
loss_q_net1, loss_q_net2, actNetloss = 0, 0, 0

policy = Policy(n_state, n_action)
for i in tqdm(range(500)):
    obs, rew = env.reset(), 0
    
    while True:
        act = policy(obs)
        next_obs, reward, done, _ = env.step(act)
        rew += reward

        update_index += 1
        if len(policy.replaybuff) > 2e3 and update_index > 4:
            update_index = 0
            loss_q_net1, loss_q_net2, actNetloss = policy.update()

        policy.replaybuff.add(obs, act, reward, next_obs, done)
        policy.replaybuff.add(obs, act, reward, next_obs, done)
        obs = next_obs
        
        if done:
            break
    if i > 0 and i % 50 == 0:
        print("itr:({:>5d}) loss:{:>3.4f} actloss:{:>3.4f} reward:{:>3.1f}".format(
            i, np.mean(loss_q_net1_list[-50:]), np.mean(loss_q_net2_list[-50:]), np.mean(loss_actNet_list[-50:]), np.mean(reward_list[-50:])))

    if policy.noise > 0.005:
        policy.noise -= (1/200)
    
    loss_q_net1_list.append(loss_q_net1), loss_q_net2_list.append(loss_q_net2), reward_list.append(rew), loss_actNet_list.append(actNetloss)


# %%
scores = [sum(run_episode(env, policy, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))

import pandas as pd
df = pd.DataFrame({'loss_q_net1': loss_q_net1_list, 'loss_q_net2': loss_q_net2_list, 'reward': reward_list, 'act loss': loss_actNet_list})
df.to_csv("./SAC.csv",
          index=False, header=True)
