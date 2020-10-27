

from tensorforce.execution import Runner


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

        states = self.env.reset()
        internals = self.agent.initial_internals()
        terminal = False
        while not terminal:
            episode_states.append(states)
            episode_internals.append(internals)
            actions, internals = self.agent.act(
                states=states, internals=internals, independent=True
            )
            episode_actions.append(actions)
            states, terminal, reward = self.env.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        self.agent.experience(
            states=episode_states, internals=episode_internals,
            actions=episode_actions, terminal=episode_terminal,
            reward=episode_reward
        )
        self.agent.update()

    def test(self):
        sum_rewards = 0.0
        for _ in range(100):
            # print(sum_rewards)
            states = self.env.reset()
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = self.agent.act(
                    states=states, internals=internals,
                    independent=True
                )
                states, terminal, reward = self.env.execute(actions=actions)
                sum_rewards += reward
        print('Mean episode reward:', sum_rewards / 100)
        self.close()

    def close(self):
        self.agent.close()
        self.env.close()











'''CustomEnvironment = or_gym.make('Knapsack-v0')
environment = Environment.create(
    environment=CustomEnvironment, max_episode_timesteps=100
)

agent=Agent.create(
    agent='a2c',
    environment=environment,
    batch_size = 10
)'''
