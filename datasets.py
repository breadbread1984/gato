#!/usr/bin/python3

import pandas as pd
import gymnasium as gym
import torch
from torch.utils.data import Dataset

def collect_episodes(model, num = 100, out_path = 'episodes.csv'):
  atari_envs = [env_id for env_id in gym.envs.registry.keys() if 'ALE/' in env_id]
  shapes = list()
  for atari_env in atari_envs:
    env = gym.make(atari_env, render_mode = 'human')
    observation, info = env.reset()
    done = False
    shapes.append(observation.shape)
  import pickle
  with open('shapes.pkl', 'wb') as f:
    f.write(pickle.dumps(shapes))

class TrajectoryDataset(Dataset):
  def __init__(self):
    super(TrajectoryDataset, self).__init__()

if __name__ == "__main__":
  collect_episodes('test')
