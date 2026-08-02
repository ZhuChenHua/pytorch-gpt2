import torch
from torchinfo import summary

from src.model import GPT2
from src.config import GPTConfig

config = GPTConfig()

gpt2 = GPT2(config)

x = torch.randint(0, config.vocab_size, (1, config.block_size))

summary(gpt2, input_data=x)
