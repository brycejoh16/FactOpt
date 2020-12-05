import numpy as np


def average_reward(training_reward: np.ndarray, K: int):
    '''
    average the training reward over K episodes
    :param K: int number of episodes to averatge over
    :return: the average reward
    '''
    mean_reward = []
    for i in np.arange(training_reward.shape[0] - K):
        mean_reward.append(np.mean(training_reward[i:i + K]))

    return np.array(mean_reward)


def hyper_parameter_search():
    print('todo')
    # todo :  make an argparser here.make a 2D grid  to do hyperparameter search.


def oracle(state: np.ndarray, N: int) -> np.ndarray:
    '''

    :param state: suggested state
    :param N: number of items
    :return: np.darray of shape[0] = N ,
    '''

    items = state[0:N]
    currency = state[N:-1]
    optimal_item = np.argmax(currency / items)
    # optimal_state = np.zeros(N)
    # optimal_state[optimal_item] = 1
    return optimal_item


