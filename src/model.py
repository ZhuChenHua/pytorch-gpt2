"""
GPT — Decoder-only Transformer 语言模型
========================================
架构（GPT-2 风格）：
  Token Embedding + Positional Embedding
  → N × Block:
        Pre-LayerNorm → Causal Multi-Head Self-Attention → 残差连接
        Pre-LayerNorm → MLP (GELU)                       → 残差连接
  → Final LayerNorm
  → LM Head（与 token embedding 权重共享）

核心设计思路：
  - Pre-norm：在每个子层前做归一化，训练更稳定
  - 权重共享：token embedding 与 LM head 共享参数，减少参数量
  - 因果掩码：F.scaled_dot_product_attention 的 is_causal=True 原生支持，
              无需手动维护下三角 mask buffer
"""

import torch.nn as nn

from src.config import GPTConfig

config = GPTConfig()


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    GPT2 模型
    模型结构：
      - Token Embedding
      - Positional Embedding
      - N × Block
      - Final LayerNorm
      - LM Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
