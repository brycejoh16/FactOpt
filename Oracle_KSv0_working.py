#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:10:36 2020

@author: diyasaha
"""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
def t(x): return torch.from_numpy(x).float()
import or_gym
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *
import tensorforce as tf

reward = []
def obj(model):
  return (-1 * (sum(model.v[i] * model.x[i] for i in range(0, model.env.N))))


def weight_rule(model):
    value = sum(model.w[i]*model.x[i] for i in range(0, model.env.N))
    cont = (0, value, 200 - model.env.state['state'][2][model.env.N])
    return(cont)

# takes in env and the index of the best choice
def oracle0(env):
    actions = []
    for r in range(0, env.N):
        ratio = env.item_values[r]/env.item_weights[r]
        actions.append(ratio)
    actions = np.array(actions)
    best_action = actions.argmax()
    reward = env.item_values[best_action]
    return (best_action,reward)

# returns the score of the actor over 2000 episodes
def score(env, actor):
    episode_rewards = []
    for i in range(2000):
        done = False
        total_reward = 0
        state = env.reset()

        while not done:
            probs = actor(t(state['state']))

            # so I am pretty sure this is action masking but I may be wrong

            # setting the zeros to tiny numbers so that backprop does not break
            action = torch.tensor(probs.detach().numpy().argmax())
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
    er = np.array(episode_rewards)
    return(np.average(er))

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)

env = or_gym.make('Knapsack-v0')
# setting up the state dims these could be wrong but they have been working so!
state_dim = 200*2 +1
n_actions = 200
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
episode_rewards = []
# making sure that the we are actually changing the env
env.randomize_params_on_reset = True
count = 0 
for i in range(50):
    done = False
    total_reward = 0
    state = env.reset()
    while not done:

        probs = actor(t(state['state']))
        pr = probs.data.numpy()
        dist = torch.distributions.Categorical(probs=probs)
        # so I am pretty sure this is action masking but I may be wrong
        mask = probs.detach().numpy() * state['action_mask']
        # setting the zeros to tiny numbers so that backprop does not break
        mask[mask == 0] = .0000001
        dist.probs = torch.tensor(mask, requires_grad=True)
        mean = np.mean(pr)
        mse = []
        for i in pr: 
            mse.append(np.square(mean - i)/len(pr))
        MSE = np.mean(mse)
        if MSE < 8e-9: 
            count = count + 1
            action = torch.from_numpy(np.array(oracle0(env)[0]))
        else:
            action = dist.sample()
  
        next_state, reward, done, info = env.step(action.detach().data.numpy())
        advantage = reward + (1 - done) * gamma * critic(t(next_state['state'])) - critic(t(state['state']))
        total_reward += reward
        state = next_state

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = -dist.log_prob(action) * advantage.detach()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()

    episode_rewards.append(total_reward)

# just making the plot a smoothed average nothing fancy here
numbers =episode_rewards
moving_averages = []
# change the window size for different levels of smoothing bigger window more smooth
window_size = 10
i =0
while i < len(numbers) - window_size + 1:
    this_window = numbers[i : i + window_size]
    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)

    i += 1
plt.scatter(np.arange(len(moving_averages)), moving_averages, s=2)
plt.title("Total reward per episode (online) masked")
plt.ylabel("reward")
plt.xlabel("episode")
plt.show()
print('% times oracle called:',(count/10000)*100)
