

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
        )

        # update the agent.
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
