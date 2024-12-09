#!/usr/bin/python3

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
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
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
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
    self.past_key_values = None
  def forward(self, inputs):
    # inputs.shape = (batch, 3, 224, 224)
    # hist.shape = (batch, hist_len, hidden)
    results = (inputs - 128.) / 128. / torch.sqrt(self.patch_size)
    results = self.conv2d(inputs) # results.shape = (batch, hidden, 7, 7)
    results = torch.flatten(results, start_dim = 2) # results.shape = (batch, hidden, 49)
    results = torch.permute(results, (0,2,1)) # results.shape = (batch, 49, hidden)
    if results.shape[1] > self.llama3.config.max_position_embeddings:
      results = results[-self.llama3.config.max_position_embeddings:]
      self.past_key_values = [(kv[0][:,:,-self.llama3.config.max_position_embeddings:,:], kv[1][:,:,-self.llama3.config.max_position_embeddings:,:]) for kv in self.past_key_values]
    attention_mask = torch.ones((results.shape[0], results.shape[1]), torch.int64) # attention_mask.shape = (batch, hist_len + 256)
    outputs = self.llama3.forward(inputs_embeds = results, attention_mask = attention_mask, past_key_values = self.past_key_values, return_dict = True, use_cache = True)
    self.past_key_values = outputs.past_key_values
    logits = outputs.last_hidden_state[:,-1,:] # logits.shape = (batch, hidden)
    action = self.pi(logits) # action.shape = (batch, 18)
    v_value = self.v_value(logits) # v.shape = (batch, 1)
    return action, v_value

if __name__ == "__main__":
  gato = Gato().to('cuda')
  inputs = torch.randn(1,3,256,256).to('cuda')
