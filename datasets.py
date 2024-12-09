#!/usr/bin/python3

import gymnasium as gym
import ale_py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def preprocess(obs):
  obs = cv2.resize(obs, (224,224)) # obs.shape = (224,224,3)
  obs = np.transpose(obs, (2,0,1)) # obs.shape = (3, 224, 224)
  return obs

def discount_cumsum(rewards, gamma = 1.):
  discount_cumsum = np.zeros_like(rewards) # discount_cumsum.shape = (len)
  discount_cumsum[-1] = rewards[-1]
  for t in reversed(range(rewards.shape[0] - 1)):
    discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
  return discount_cumsum

def generate_trajectories(trajectories = 10, policy = None, seed = None):
  gym.register_envs(ale_py)
  # pick a Atari env
  for env_id in gym.envs.registry.keys():
    if not ('ALE/' in env_id and 'ram' not in env_id): continue
    env = gym.make(env_id, render_mode = "rgb_array")
    # collect trajectories
    trajectories = list()
    for _ in range(trajectories):
      states, rewards, actions, dones, returns = list(), list(), list(), list(), list()
      obs, info = env.reset(seed = seed)
      states.append(preprocess(obs)) # s_t
      while True:
        if policy is None:
          action = env.action_space.sample()
        else:
          # TODO: use policy
          pass
        obs, reward, done, trunc, info = env.step(action)
        rewards.append(reward) # r_t
        actions.append(action) # a_t
        dones.append(done)
        returns.append(sum(rewards))
        if done:
          assert len(states) == len(actions) == len(rewards) == len(dones)
          trajectories.append({'observations': np.stack(states, axis = 0), # shape = (len, 3, 224, 224)
                               'actions': np.stack(actions, axis = 0), # shape = (len)
                               'rewards': np.stack(rewards, axis = 0), # shape = (len)
                               'dones': np.stack(dones, axis = 0), # shape = (len)
                               'returns': np.stack(returns, axis = 0)}) # shape = (len)
          trajectories[-1]['rtg'] = discount_cumsum(trajectories[-1]['rewards']) # V(s_t)
          break
        states.append(preprocess(obs)) # s_{t+1}
    env.close()
  return trajectories

class TrajectoryDataset(Dataset):
  def __init__(self):
    super(TrajectoryDataset, self).__init__()

if __name__ == "__main__":
  trajectories = generate_trajectories(1)
  import pdb; pdb.set_trace()
