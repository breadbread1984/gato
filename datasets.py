#!/usr/bin/python3

import pandas as pd
from multiprocessing.pool import ThreadPool
import gymnasium as gym
import torch
from torch.utils.data import Dataset

def get_shape(atari_env):
  env = gym.make(atari_env, render_mode = 'human')
  observation, info = env.reset()
  action = env.action_space.sample()
  observation, reward, done, _, info = env.step(action)
  env.close()
  return observation.shape

def collect_episodes(model, num = 100, out_path = 'episodes.csv'):
  atari_envs = [env_id for env_id in gym.envs.registry.keys() if 'ALE/' in env_id and 'ram' not in env_id]
  print(atari_envs)
  pool = ThreadPool(processes = 10)
  results = pool.map_async(get_shape, atari_envs)
  shapes = results.get()
  import pickle
  with open('shapes.pkl', 'wb') as f:
    f.write(pickle.dumps(shapes))

class TrajectoryDataset(Dataset):
  def __init__(self):
    super(TrajectoryDataset, self).__init__()

if __name__ == "__main__":
  collect_episodes('test')
