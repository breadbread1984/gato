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
  obs = np.expand_dims(obs, axis = 0) # obs.shape = (1,3,224,224)
  return obs

def discount_cumsum(rewards, gamma = 1.):
  discount_cumsum = np.zeros_like(rewards) # discount_cumsum.shape = (len)
  discount_cumsum[-1] = rewards[-1]
  for t in reversed(range(rewards.shape[0] - 1)):
    discount_cumsum[t] = rewards[t] + gamma * discount_cumsum[t + 1]
  return discount_cumsum

def gae(rewards, values, dones, gamma, lam):
  T = len(rewards)
  advantages = np.zeros(T)
  gae_ = 0
  for t in reversed(range(T)):
    delta = rewards[t] + (gamma * values[t + 1] if not dones[t] else 0) - values[t]
    advantages[t] = delta + (gamma * lam * advantages[t + 1] if not dones[t] else 0)
  return advantages

if __name__ == "__main__":
  rewards = np.random.normal(size = (10,))
  values = np.random.normal(size = (10,))
  dones = np.concatenate([np.zeros(9), np.ones(1)], axis = 0)
  gamma = 0.95
  lam = 0.95
  g = gae(rewards, values, dones, gamma, lam)
  print(g)
