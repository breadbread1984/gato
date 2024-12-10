#!/usr/bin/python3

from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import Cache, DynamicCache
from transformers.models.llama import LlamaModel, LlamaConfig

def create_llama3_8b():
  # vocab_size is size of action space 18 + bos + eos
  config = {"vocab_size": 20,
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
            "mlp_bias": False}
  return Gato(config)

def create_llama3_0_5b():
  config = {"vocab_size": 20,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-06,
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
            "mlp_bias": False}
  return Gato(config)

class Gato(nn.Module):
  def __init__(self, llama_config = {
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
    self.encoder = torchvision.models.resnet18(pretrained = True)
    self.encoder.fc = nn.Linear(512, self.llama3.config.hidden_size)
    self.pi = nn.Linear(self.llama3.config.hidden_size, 18)
    self.v_value = nn.Linear(self.llama3.config.hidden_size, 1)
  def forward(self, inputs, past_key_values = DynamicCache()):
    assert isinstance(past_key_values, Cache)
    # inputs.shape = (batch, 3, 224, 224)
    # past_key_values.shape = (layer_num, 2, batch, head, seq_len, hidden / head)
    results = (inputs - 128.) / 128. # results.shape = (batch, 3, 224, 224)
    results = self.encoder(results) # results.shape = (batch, hidden)
    results = torch.unsqueeze(results, dim = 1) # results.shape = (batch, 1, hidden)
    seq_length = past_key_values.get_seq_length()
    attention_mask = torch.ones((results.shape[0], seq_length + results.shape[1]), dtype = torch.int64).to(next(self.parameters()).device) # attention_mask.shape = (batch, seq_len + 1)
    outputs = self.llama3.forward(inputs_embeds = results, attention_mask = attention_mask, past_key_values = past_key_values, use_cache = True)
    logits = outputs.last_hidden_state[:,-1,:] # logits.shape = (batch, hidden)
    action = torch.softmax(self.pi(logits), dim = -1) # action.shape = (batch, 18)
    v_value = self.v_value(logits) # v.shape = (batch, 1)
    return action, v_value, outputs.past_key_values

if __name__ == "__main__":
  gato = Gato().to('cuda')
  inputs = torch.randn(1,3,256,256).to('cuda')
