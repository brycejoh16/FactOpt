
import numpy as np
from tensorforce.environments import Environment
import or_gym
class KnapSackEnv(Environment):
    def __init__(self):
        '''
        KnapSack environment defined for usage with tensorforce
        '''
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        return dict(type='int', num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; restrict training timesteps via
    #     Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = np.random.random() < 0.5
        reward = np.random.random()
        return next_state, terminal, reward

def knapsack_env():
    env=or_gym.make('Knapsack-v0')

    # todo: take constraints from or_gym created and put into
    #  class knapsack environment.
    raise ValueError('This still has to be done!')
