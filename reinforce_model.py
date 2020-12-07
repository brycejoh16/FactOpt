



import or_gym
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os ,sys
import search_modules as sm
import plot_modules as pm
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
            # nn.ReLU(),
            nn.Softmax(dim=0)
        )

    def forward(self,x):
        return self.model(x)

class critic(nn.Module):
    def __init__(self,N):
        super(critic, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(2*N+1,1)
        )
    def forward(self,x):
        return self.model(x)

def train(env:or_gym.envs.classic_or,net:actor,
          net2:critic,
          nb_episodes:int,optimizer:torch.optim,optimizer_critic:torch.optim,
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
    CL = []
    AL = []
    for j in range(nb_episodes):
        env.reset()
        state=env.state
        done=False
        eps_reward=0
        i=0
        # print('episode %i'%j)


        criterion=nn.MSELoss()
        actor_loss=torch.tensor([0],dtype=torch.float,requires_grad=True)
        critic_loss=torch.tensor([0],dtype=torch.float,requires_grad=True)

        while not done:
            out=net(numpy2torch(state['state']))
            dist = torch.distributions.Categorical(probs=out)

            # sample action a_t
            a_t = dist.sample()
            idx=int(np.mean(torch2numpy(a_t)))

            next_state, reward, done,_=env.step(idx)

            # right now just consider a value function based off of the state
            # then consider others.
            v0 = net2(numpy2torch(state['state']))
            v1= net2(numpy2torch(next_state['state']))
            # calculate the temporal difference target
            td_target=reward + discout* v1
            td_error=reward + discout*v1 - v0


            # implement the least squares loss for the critic
            critic_loss=critic_loss.detach()+criterion(v1,td_target).reshape(1)
            # implement the log probability loss for the actor
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()


            actor_loss = actor_loss.detach() + dist.log_prob(a_t) * td_error.detach()

            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()


            eps_reward += reward
            state=next_state
            i+=1

            # print(state)
            # print(reward)
            # print(done)


        # out_batch=torch.reshape(torch.cat(OUT,0),(i,N))
        # define loss function for jsut the actor

        CL.append(torch2numpy(critic_loss)[0])
        AL.append(torch2numpy(actor_loss)[0])
        episode_nb_steps.append(i)
        episode_reward.append(eps_reward)

    return np.array(episode_reward),np.array(episode_nb_steps),np.array(CL),np.array(AL)


# def test(env:or_gym.envs.classic_or,net:actor,
#           nb_episodes:int,optimizer:torch.optim,
#           discout:float,
#           loss):

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
    net2=critic(N)
    print(net)
    print(net2)

    K=7
    nb_episodes = 10000

    LR=np.array([ 10**i for i in np.arange(-2,3,1,dtype='float')])
    CLR = np.array([10 ** i for i in np.arange(-2, 3, 1, dtype='float')])

    directory = 'linear_actor,linear_critic,one_step,actor_critic_episodes_%i_items_%i' % (nb_episodes, N)
    failure = os.system('mkdir ./results/' + directory)
    discout=0.99
    MR=[]
    for lr in LR:
        mr=[]
        for clr in CLR:

            print('alr :%0.4f , clr:%0.4f'%(lr,clr))
            prefix = 'alr_%0.4f_clr_%0.4f' % (lr, clr)
            optimizer = optim.Adam(net.parameters(), lr=lr)
            optimizer_critic=optim.Adam(net2.parameters(),lr=clr)
            training_reward,eps_steps,CL,AL=train(env,net,net2,discout=discout,
                                            nb_episodes=nb_episodes,optimizer=optimizer,optimizer_critic=optimizer_critic)
            mean_reward=sm.average_reward(training_reward,K=K)
            mr.append(np.mean(mean_reward))
            pm.plot_train_reward(training_reward,mean_reward,eps_steps,title='Average Reward: %0.2f'%mr[-1],prefix=prefix,
                              K=K,direcotry=directory)
            pm.plot_loss(AL,CL,directory,prefix,K=K)

        MR.append(np.array(mr))

    pm.plot_mean_rewards(MR,LR,CLR,directory=directory)





