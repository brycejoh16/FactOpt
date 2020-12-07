import matplotlib.pyplot as plt
import numpy as np
import search_modules as sm

from matplotlib.ticker import MaxNLocator
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
    plt.plot(eps[0:mean_reward.shape[0]],mean_reward,label='smoothed reward over K:%i episodes'%K)
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


def plot_loss(AL:np.ndarray,CL:np.ndarray,directory:str,prefix:str,K:int):
    eps = np.arange(AL.shape[0])
    dir = './results/' + directory
    fig, ax1 = plt.subplots(1, 1, figsize=[5, 3], dpi=300)
    fig.suptitle('loss functions for K:%i smoothing'%K)
    color = 'tab:red'
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('actor loss', color=color)
    avg_loss_a=sm.average_reward(AL,K=K)
    ax1.plot(eps[0:avg_loss_a.shape[0]],avg_loss_a, color=color)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('critic loss', color=color)  # we already handled the x-label with ax1
    avg_loss_c=sm.average_reward(CL,K=K)
    ax2.plot(eps[0:avg_loss_c.shape[0]],avg_loss_c, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(dir + '/%s_loss.png' % prefix)
    plt.close(fig)
def plot_mean_rewards(MR:list,LR:np.ndarray,CLR:np.ndarray,directory:str):
    for lr,mr in zip(LR,MR):
        plt.semilogx(CLR,mr,label='alr: %0.2f'%lr)
    plt.legend()
    plt.title('Mean reward for each learning rate')
    plt.xlabel('critic learning rate')
    plt.ylabel('mean reward over all episodes')
    dir = './results/' + directory
    plt.savefig(dir+'/analysis_of_learning_rates.png')

