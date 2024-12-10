#!/usr/bin/python3

from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.llama import LlamaModel, LlamaConfig

def create_llama3_8b():
  # vocab_size is size of action space 18 + bos + eos
  config = LlamaConfig(vocab_size = 20,
                       hidden_size = 4096,
                       intermediate_size = 14336,
                       num_hidden_layers = 24,
                       num_attention_heads = 32,
                       num_key_value_heads = 8,
                       hidden_act = "silu",
                       max_position_embeddings = 4096,
                       initializer_range = 0.02,
                       rms_norm_eps = 1e-05,
                       use_cache = True,
                       pad_token_id = None,
                       bos_token_id = 18,
                       eos_token_id = 19,
                       pretraining_tp = 1,
                       tie_word_embeddings = False,
                       rope_theta = 500000.,
                       rope_scaling = None,
                       attention_bias = False,
                       attention_dropout = 0.,
                       mlp_bias = False)
  return LlamaForCausalLM(config)

class Gato(nn.Module):
  def __init__(self, patch_size = 32, llama_config = {
    "vocab_size": 20,
    "hidden_size": 1024,
    "intermediate_size": 14336,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "initializer_range": 0.02, 
    "rms_norm_eps": 1e-05,
    "use_cache": True,
    "pad_token_id": None,
    "bos_token_id": 18,
    "eos_token_id": 19,
    "pretraining_tp": 1,
    "tie_word_embeddings": False,
    "rope_theta": 500000., 
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.,
    "mlp_bias": False}):
    super(Gato, self).__init__()
    self.llama3 = LlamaModel(LlamaConfig(**llama_config))
    self.conv2d = nn.Conv2d(3, self.llama3.config.hidden_size, kernel_size = patch_size, stride = patch_size, padding = 0)
    self.pi = nn.Linear(self.llama3.config.hidden_size, 18)
    self.v_value = nn.Linear(self.llama3.config.hidden_size, 1)
    self.patch_size = patch_size
  def forward(self, inputs, past_key_values = DynamicCache()):
    # inputs.shape = (batch, 3, 224, 224)
    # past_key_values.shape = (layer_num, 2, batch, head, seq_len, hidden / head)
    results = (inputs - 128.) / 128. / np.sqrt(self.patch_size)
    results = self.conv2d(results) # results.shape = (batch, hidden, 7, 7)
    results = torch.flatten(results, start_dim = 2) # results.shape = (batch, hidden, 49)
    results = torch.permute(results, (0,2,1)) # results.shape = (batch, 49, hidden)
    seq_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    attention_mask = torch.ones((results.shape[0], seq_length + results.shape[1]), dtype = torch.int64).to(next(self.parameters()).device) # attention_mask.shape = (batch, 49)
    outputs = self.llama3.forward(inputs_embeds = results, attention_mask = attention_mask, past_key_values = past_key_values, use_cache = True)
    logits = outputs.last_hidden_state[:,-1,:] # logits.shape = (batch, hidden)
    action = torch.softmax(self.pi(logits), dim = -1) # action.shape = (batch, 18)
    v_value = self.v_value(logits) # v.shape = (batch, 1)
    return action, v_value, outputs.past_key_values

if __name__ == "__main__":
  gato = Gato().to('cuda')
  inputs = torch.randn(1,3,256,256).to('cuda')
