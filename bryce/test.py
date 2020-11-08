

# this is a test file
import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt


def t(x): return torch.from_numpy(x).float()


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
def main():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)


    # training hyperparameters right here.
    # here we are constructing an optimizer object
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # future reward discount... hyperparameter tuned in some way .
    # future reward discount: sums up value of all future states.
    # [0,1] how greedy are you going to be.
    gamma = 0.99

    episode_rewards = []

    for i in range(500):
        print('episode %i'%i)
        done = False
        total_reward = 0
        state = env.reset()
        # monte carlo random walk
        # done :boolean a terminal condition.
        while not done:
            # find the probabilities
            probs = actor(t(state))

            # make a categorical ditribution from probailities
            dist = torch.distributions.Categorical(probs=probs)

            # randomly sample from that. distribution is built on prior information.
            action = dist.sample()

            # environment takes its next step. , actor critic stuff.
            next_state, reward, done, info = env.step(action.detach().data.numpy())

            # COULD BE: part of tuning system ... but we dont know yet.
            # gamma being closer to zero means more greedy.

            # so accounting for future state more. #todo: come back to this
            advantage = reward + (1 - done) * gamma * critic(t(next_state)) - critic(t(state))


            total_reward += reward
            state = next_state


            # critic loss ... mean(a^2)
            # we are optimizing the hyperparameters of the nn's


            # todo: look at the optimization parameters.
            critic_loss = advantage.pow(2).mean() # why are they doing a mean call here?
            adam_critic.zero_grad()

            # back propagation
            critic_loss.backward()

            # Performs a single optimization step (parameter update).
            adam_critic.step()

            actor_loss = -dist.log_prob(action) * advantage.detach()
            adam_actor.zero_grad()
            actor_loss.backward()
            adam_actor.step()

        episode_rewards.append(total_reward)

    plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
    plt.title("Total reward per episode (online)")
    plt.ylabel("reward")
    plt.xlabel("episode")
    plt.show()

if __name__ == '__main__':
    main()