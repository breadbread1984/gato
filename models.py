#!/usr/bin/python3

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama import LlamaForCausalLM, LlamaConfig

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
  def __init__(self, ):
    self.llama3 = create_llama3_8b()
    super(Gato, self).__init__()
  def forward(self, inputs):
    # NOTE: inputs
    pass

if __name__ == "__main__":
  llama3 = create_llama3_8b()
