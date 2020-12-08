<<<<<<< Updated upstream


from tensorforce.execution import Runner
from tensorforce.agents import ActorCritic
from tensorforce.agents import AdvantageActorCritic


#initialize environment using .make() or-gym. Environment.create wraps custom
#environments' methods (I think) to work with tensorforce api

class RL():
    def __init__(self,agent=None,env=None):
        '''
        initilizer for reinforcement learning module
        :param agent: string keyword of registered agent from tensorforce [default : a2c] the
        advantage actor critic
        :param env: Custom environment class which tells
        '''
        self.agent=agent
        self.env=env
    def train(self,nb_episodes=100):
        '''
        training function for this instance of RL
        :param nb_episodes: number of epidsodes to run in the train
        :return:
        '''
        for _ in range(nb_episodes):
            self.episode()
        #todo: we should have some parameter here stating how well the train went.
    def episode(self):
        '''
        a single episode ... todo :describe what this does here...basically what is an episode
        :return:
        '''
        episode_states = list()
        episode_internals = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()


        # at start of an environment start at a random state.
        # start an monte carlo random walk by starting at a random state.
        states = self.env.reset()

        # i don't know what internals do yet...
        internals = self.agent.initial_internals()

        # our environment class will tell us when we want to stop
        terminal = False
        while not terminal:
            # this is a MONTE CARLO RANDOM WALK..
            # this walk is dependent on the condition of
            # terminal which we specified by the environment.
            # the terminal condition is specified by the custom environment class
            # in the execute function.
            episode_states.append(states)
            episode_internals.append(internals)

            # here is where the agent acts, this says which action should i take?
            # this is merely a suggested action ... 
            actions, internals = self.agent.act(
                states=states, internals=internals, independent=True
            )
            episode_actions.append(actions)


            # here is where the environment executes... in custom environment we will
            # have to define this function.
            states, terminal, reward = self.env.execute(actions=actions)

            # if the reward look similar from previous steps , and is BAD,
            # in this monte carlo random walk, we know we are stuck
            # and need to call the oracle.

            # todo: CALL ORACLE HERE.  but still need to update internals somehow. b/c
            #  we will have one more state than internal.
            #  so probabably have to call states, internals again...

            episode_terminal.append(terminal)
            episode_reward.append(reward)


        # update the state of the agent based on what it just learned in the previous episode.
        self.agent.experience(
            states=episode_states, internals=episode_internals,
            actions=episode_actions, terminal=episode_terminal,
            reward=episode_reward
=======
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
            nn.Softmax(dim=0)
>>>>>>> Stashed changes
        )

    def forward(self, X):
        return self.model(X)


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

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99

episode_rewards = []

for i in range(500):
    done = False
    total_reward = 0
    state = env.reset()

    while not done:
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        next_state, reward, done, info = env.step(action.detach().data.numpy())
        advantage = reward + (1 - done) * gamma * critic(t(next_state)) - critic(t(state))

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

plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
plt.title("Total reward per episode (online)")
plt.ylabel("reward")
plt.xlabel("episode")
plt.show()