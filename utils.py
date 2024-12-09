#!/usr/bin/python3

from tqdm import tqdm
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
    discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
  return discount_cumsum

def gae(diffs, lambda = 1.):
  powers = torch.range(diffs.shape[0]).to(diffs.device)
  weighted = torch.pow(diffs, powers)
  gae = torch.sum(weighted)
  return gae

if __name__ == "__main__":
  trajectories = generate_trajectories(1)
  import pdb; pdb.set_trace()
