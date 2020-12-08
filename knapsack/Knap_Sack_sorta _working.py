import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def t(x): return torch.from_numpy(x).float().to(device)
import gym
import or_gym
# Actor module, categorical actions only


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=0)
        ).to(device)

    def forward(self, X):
        return self.model(X)

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=0)
        ).to(device)
    def forward(self, X):
        return self.model(X)

env = gym.make('Knapsack-v0')
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

for i in range(1000):
    if i == 100 or i == 250 or i == 500 or i == 900:
        print(i)
    done = False
    total_reward = 0
    state = env.reset()
    while not done:
        probs = actor(t(state['state']))
        dist = torch.distributions.Categorical(probs=probs)
        # so I am pretty sure this is action masking but I may be wrong
        mask = probs.detach().cpu().numpy() * state['action_mask']
        # setting the zeros to tiny numbers so that backprop does not break
        mask[mask == 0] = .0000001
        dist.probs = torch.tensor(mask, requires_grad=True)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.detach().data.to(device))
        advantage = reward + (1 - done) * gamma * critic(t(next_state['state'])) - critic(t(state['state']))

        total_reward += reward
        state = next_state

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = -dist.log_prob(action) * advantage.detach()
        adam_actor.zero_grad()
        actor_loss.sum().backward()
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
