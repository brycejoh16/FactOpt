

import or_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os ,sys
import search_modules as sm
def numpy2torch(a:np.ndarray,long=False):
    if long:
        return torch.from_numpy(a).long()
    else:
        return torch.from_numpy(a).float()

def torch2numpy(a:torch.Tensor):
    return a.detach().numpy()

class actor(nn.Module):
    def __init__(self,N):
        super(actor,self).__init__()
        # simple sequential nueral network
        self.model=nn.Sequential(
            nn.Linear(2*N+1,N),
            nn.ReLU()
            # nn.Softmax()
        )

    def forward(self,x):
        return self.model(x)


def train(env:or_gym.envs.classic_or,net:actor,
          nb_episodes:int,optimizer:torch.optim,
            criterion,
          discout:float=0.1,
          )-> (np.ndarray,np.ndarray):
    '''
    this is the train function
    :param env: knapsack environment object
    :param net: actor for linear model
    :param nb_episodes: number of episodes to execute for infinite horizon problem
    :param optimizer: torch.optim
    :return: training 2d ndarray with
    '''
    N=env.N
    episode_reward = []
    episode_nb_steps=[]
    for j in range(nb_episodes):
        env.reset()
        state=env.state
        done=False
        eps_reward=0
        i=0
        print('episode %i'%j)
        Target=[]
        OUT=[]
        while not done:
            # print(i)
            # the first state
            out=net(numpy2torch(state['state']))
            # print(out)
            # take a step in the environment
            state, reward, done,_=env.step(np.argmax(torch2numpy(out)))
            eps_reward+=reward

            # print(state)
            # print(reward)
            # print(done)


            optimal_state=sm.oracle(state['state'],N)
            #this is easy optimization, we compare the ideal target to the expected
            # target in and L2 norm optimization
            Target.append(optimal_state)
            OUT.append(out)
            i+=1

            # print('current state:')
            # print(state['state'])
            # print('eps reward')
            # print(eps_reward)
            # print('# of steps: %i'%i)


        out_batch=torch.reshape(torch.cat(OUT,0),(i,N))
        target_batch=torch.tensor(Target,dtype=torch.long)
        loss = criterion(out_batch,target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_nb_steps.append(i)
        episode_reward.append(eps_reward)

    return np.array(episode_reward),np.array(episode_nb_steps)


# def test(env:or_gym.envs.classic_or,net:actor,
#           nb_episodes:int,optimizer:torch.optim,
#           discout:float,
#           loss):
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



if __name__=='__main__':

    N=10
    # print(net)
    # param=list(net.parameters())
    # print(param[0].size())

    env_config={ 'N':N,
                 'max_weight': 200,
                 'current_weight': 0,
                 'mask':True,
                 'randomize_params_on_reset': True,
    }

    env=or_gym.make('Knapsack-v0',env_config=env_config)
    # set seed for reproducibility
    # right now take fresh entropy from the computer
    env.set_seed(int.from_bytes(os.urandom(4), sys.byteorder))

    net = actor(N)
    #print(net)

    K=7
    nb_episodes = 10000
    criterion = nn.CrossEntropyLoss()
    directory = 'relu_only_actor,CrossEntropy,oracletrainer,batch_updates'

    LR=np.array([ 10**i for i in np.arange(-2,3,1,dtype='float')])


    failure=os.system('mkdir ./results/'+directory)


    for lr in LR:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        training_reward,eps_steps=train(env,net,nb_episodes=nb_episodes,optimizer=optimizer,criterion=criterion)
        mean_reward=sm.average_reward(training_reward,K=K)
        plot_train_reward(training_reward,mean_reward,eps_steps,title='With oracle trainer',prefix='oracle_trainer_episodes_%i_items_%i_lr_%0.8f_relu'%(nb_episodes,N,lr),
                          K=K,direcotry=directory)









