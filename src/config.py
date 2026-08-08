import os
from dataclasses import dataclass

# 在项目根目录设置缓存目录路径
cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)


@dataclass
class GPTConfig:
    """GPT 模型架构超参数。"""

    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens in the vocabulary: 50000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


@dataclass
class TrainConfig:
    """预训练超参数（src/pretrain.py 使用）。"""

    batch_size: int = 4  # 每个 batch 的序列数
    learning_rate: float = 3e-4
    max_iters: int = 1000  # 总训练步数
    eval_iters: int = 10  # 每次评估在 train/val 上各取多少个 batch 求平均 loss
    eval_interval: int = 500  # 每多少步评估一次 train / val loss
    sample_every: int = 200  # 每多少步打印一段生成的文本（训练中多次触发，直观看到学习进度）
    weight_decay: float = 0.1  # 权重衰减（仅作用于 2D 权重）
    out_dir: str = os.path.join("model", "shakespeare")  # checkpoint 保存目录
