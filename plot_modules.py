import matplotlib.pyplot as plt
import numpy as np


def plot_train_reward(training_reward:np.ndarray,mean_reward:np.ndarray ,eps_nb_steps:np.ndarray,prefix:str,
                      title:str,K:int,direcotry:str):
    '''

    :param training_reward:
    :param eps_nb_steps:
    :param prefix:
    :param title:
    :return:
    '''
    # mean_reward= np.mean(training_reward,axis=1)
    # max_reward=np.max(training_reward,axis=1)
    # min_reward=np.min(training_reward,axis=1)
    dir='./results/'+direcotry
    # plot the results
    eps=np.arange(training_reward.shape[0])
    plt.title(title)
    plt.scatter(eps,training_reward,label='total-reward per episode',s=0.3)
    plt.plot(eps[0:mean_reward.shape[0]],mean_reward,label='mean reward over K:%i average'%K)
    plt.xlabel('episodes')
    plt.legend()
    plt.ylabel('reward')
    plt.savefig(dir+'/%s_rewards.png'%prefix)
    plt.close()

    plt.plot(eps,eps_nb_steps)
    plt.xlabel('episode')
    plt.ylabel('steps per episode')
    plt.savefig(dir+'/%s_steps.png'%prefix)
    plt.close()
