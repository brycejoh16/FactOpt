
from tensorforce.agents import Agent
import knapsack.reinforcement_learning as rl
import knapsack.environments as E
import time
def driver():
    '''
    This is the driver function which will allow us to use multiple drive
    :return:
    '''
    # first need to instantiate an RL class , and make an instance
    # of the knapsack environment. How this is defined is shown in environments.KnapSackEnv()
    #
    env=E.knapsack_env()
    agent=Agent.create(
            agent='a2c',
            environment=env,
            batch_size=10
        )

    ks= rl.RL(agent=agent,env=env)
    ks.train(nb_episodes=100)
    ks.test()

    # now we need to look at the results somehow, prefarably through like tensorboard

    # after we have instantiated we want  want to


if __name__=='__main__':
    driver()