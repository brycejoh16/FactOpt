import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import *
import numpy as np
reward = []
import torch

def obj(model):
  return (-1 * (sum(model.v[i] * model.x[i] for i in range(0, model.env.N))))


def weight_rule(model):
    value = sum(model.w[i]*model.x[i] for i in range(0, model.env.N))
    cont = (0, value, 200 - model.env.state['state'][2][model.env.N])
    return(cont)

# takes in a env and returns list of optimal actions
def oracle1(env):
  model = pyo.ConcreteModel()

  model.w = env.item_weights
  model.v = env.item_values

  model.env = env
  model.x = pyo.Var(range(0, env.N), within=pyo.Binary)


  model.OBJ = pyo.Objective(rule=obj)

  model.weight_rule = Constraint(rule=weight_rule)

  solver_manager = pyo.SolverManagerFactory('neos')
  results = solver_manager.solve(model, opt='cplex')
  model.pprint()

  actions = []
  for r in range(0, env.N):
      if env.state['avail_actions'][r]*pyo.value(model.x[r]) ==1:
          actions.append(r)

  return actions

# takes in env and the index of the best choice
def oracle0(env):
    actions = []
    for r in range(0, env.N):
        ratio = env.item_values[r]/env.item_weights[r]
        actions.append(ratio)
    actions = np.array(actions)
    return actions.argmax()

# returns the score of the actor over 2000 episodes
def score(env, actor):
    episode_rewards = []
    for i in range(2000):
        done = False
        total_reward = 0
        state = env.reset()

        while not done:
            probs = actor(t(state['state']))

            # so I am pretty sure this is action masking but I may be wrong

            # setting the zeros to tiny numbers so that backprop does not break
            action = torch.tensor(probs.detach().numpy().argmax())
            next_state, reward, done, info = env.step(action.detach().data.numpy())
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
    er = np.array(episode_rewards)
    return(np.average(er))
