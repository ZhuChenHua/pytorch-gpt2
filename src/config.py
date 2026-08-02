import os
from dataclasses import dataclass

# 在项目根目录设置缓存目录路径
cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)


@dataclass
class GPTConfig:
    """GPT 超参数配置。"""

    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens in the vocabulary: 50000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
