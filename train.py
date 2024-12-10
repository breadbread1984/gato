#!/usr/bin/python3

import gymnasium as gym
import ale_py
from absl import flags, app
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, device
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from models import Gato
from utils import preprocess, discount_cumsum, gae

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('epochs', default = 500, help = 'epochs')
  flags.DEFINE_integer('seed', default = None, help = 'seed')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'checkpoint')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma')
  flags.DEFINE_float('lam', default = 0.95, help = 'lambda')

def main(unused_argv):
  policy = Gato().to(device(FLAGS.device))
  optimizer = Adam(policy.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  criterion = nn.MSELoss()
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  gym.register_envs(ale_py)
  env_ids = [env_id for env_id in gym.envs.registry.keys() if 'ALE/' in env_id and 'ram' not in env_id]
  global_steps = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = torch.load(join(FLAGS.ckpt, 'model.pth'))
    global_steps = ckpt['global_steps']
    policy.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
  for i in tqdm(range(FLAGS.epochs)):
    for env_id in env_ids:
      optimizer.zero_grad()
      env = gym.make(env_id, render_mode = "rgb_array")
      states, rewards, actions, v_preds, dones = list(), list(), list(), list(), list()
      past_key_values = None
      obs, info = env.reset(seed = FLAGS.seed)
      states.append(preprocess(obs)) # s_t
      while True:
        inputs = torch.from_numpy(np.stack(states, axis = 0)).to(next(policy.parameters()).device) # inputs.shape = (len, 224, 224, 3)
        action, v_pred, past_key_values = policy(inputs, past_key_values = past_key_values) # action.shape = (batch, 18), v_pred.shape = (batch, 1)
        # run in environment
        act = torch.argmax(action[0], dim = 0) # act.shape = ()
        obs, reward, done, info = env.step(act.detach().cpu().numpy())
        rewards.append(torch.Tensor(reward).to(next(policy.parameters()).device)) # r_t.shape = ()
        actions.append(action[0]) # a_t.shape = (18,)
        v_preds.append(v_pred[0]) # hat{V}(s_t).shape = (1)
        dones.append(done)
        if done:
          assert len(states) == len(actions) == len(rewards) == len(v_preds)
          observations = np.stack(states, axis = 0) # shape = (len, 3, 224, 224)
          actions = torch.stack(actions, axis = 0) # shape = (len)
          rewards = torch.stack(rewards, axis = 0) # shape = (len)
          v_trues = discount_cumsum(rewards, gamma = FLAGS.gamma) # V(s_t).shape = (len)
          v_preds = torch.cat(v_preds, dim = 0) # v_preds.shape = (len)
          advantages = gae(rewards, v_preds, dones, FLAGS.gamma, FLAGS.lam)
          break
        states.append(preprocess(obs)) # s_{t+1}
      probs = torch.stack(actions, dim = 0) # actions.shape = (len, 18)
      logprobs = torch.max(torch.log(probs), dim = -1) # log_prob.shape = (len)
      loss = - torch.mean(logprobs * advantages) + 0.5 * criterion(v_preds, v_trues)
      loss.backward()
      optimizer.step()
      env.close()
      tb_writer.add_scalar('loss', loss, global_steps)
      global_steps += 1
    scheduler.step()
    ckpt = {
      'global_steps': global_steps,
      'state_dict': policy.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(ckpt, join(FLAGS.ckpt, 'model.pth'))

if __name__ == "__main__":
  add_options()
  app.run(main)

