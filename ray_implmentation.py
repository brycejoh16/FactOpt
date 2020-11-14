
import or_gym
import numpy as np
import ray
from ray.rllib import agents
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from gym import spaces
from or_gym.utils import create_env
import pandas as pd
import time
import matplotlib.pyplot as plt
tf,_,_= try_import_tf()


class KP0ActionMaskModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name, true_obs_shape=(11,),
                 action_embed_size=5, *args, **kwargs):
        super(KP0ActionMaskModel, self).__init__(obs_space,
                                                 action_space, num_outputs, model_config, name,
                                                 *args, **kwargs)

        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape),
            action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})


        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector,
                                      axis=1)
        # note i changed some of this
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


def register_env(env_name, env_config):
    env = create_env(env_name)
    tune.register_env(env_name, lambda env_name: env(env_name, env_config=env_config))



if __name__=='__main__':
    env = or_gym.make('Knapsack-v0')
    print("Max weight capacity:\t{}kg".format(env.max_weight))
    print("Number of items:\t{}".format(env.N))

    env_config = {'N': 5,
                  'max_weight': 15,
                  'item_weights': np.array([1, 12, 2, 1, 4]),
                  'item_values': np.array([2, 4, 2, 1, 10]),
                  'mask': True}

    env = or_gym.make('Knapsack-v0', env_config=env_config)

    print("Max weight capacity:\t{}kg".format(env.max_weight))
    print("Number of items:\t{}".format(env.N))

    print(env.state)

    # state, reward, done, _ = env.step(1)
    # print(state)
    ModelCatalog.register_custom_model('kp_mask', KP0ActionMaskModel)
    register_env('Knapsack-v0', env_config=env_config)
    if ray.is_initialized():
        ray.shutdown()

    ray.init(ignore_reinit_error=True)

    trainer_config = {
        "model": {
            "custom_model": "kp_mask"
        },
        "env_config": env_config,
        'num_workers': 3,
    }
    # initilize the trainer module
    trainer = agents.ppo.PPOTrainer(env='Knapsack-v0', config=trainer_config)


    results = []
    episode_data = []
    episode_json = []
    N=1000
    T=[]
    for n in range(N):
        t=time.time()
        result = trainer.train()
        end=time.time()- t
        results.append(result)

        episode = {'n': n,
                   'episode_reward_min': result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max': result['episode_reward_max'],
                   'episode_len_mean': result['episode_len_mean']}

        episode_data.append(episode)
        T.append(end)

        if n %50 ==0  and n >49 or n==0:
            print(n)
            print(T[-1])
            print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}'
                  f'/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')

            df = pd.DataFrame(data=episode_data)
            plt.scatter(np.arange(n+1), df['episode_reward_mean'])
            plt.savefig('./trialruns/run_nb_episodes_%i.png'%n)

    # env = trainer.env_creator('Knapsack-v0')
    # state = env.state
    # state['action_mask'][0] = 0


    # trainer.train()

