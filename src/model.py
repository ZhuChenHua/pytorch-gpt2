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

import math

import torch
import torch.nn as nn

from src.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention 模块：
    将输入矩阵进行线性变换，得到查询（query）、键（key）和值（value）矩阵，然后计算注意力权重，并将其应用于值矩阵，最后再进行线性变换得到输出。该模块使用了因果掩码，确保每个位置只能关注到它之前的位置，从而实现自回归建模。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # query, key, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 'bias' buffer is a lower triangular matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # B = batch size, T = sequence length, C = embedding dimension
        B, T, C = x.size()
        # reshape x -> (B, T, 3*C)
        qkv = self.c_attn(x)
        # (B, T, 3*C) -> (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh,T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh,T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh,T, hs)

        # Dot-Product Attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        att = torch.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh,T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    MLP 模块：
    对输入进行线性变换，然后应用 GELU 激活函数，最后再进行线性变换。其中，GELU 激活函数是一种非线性函数，可以更好地捕捉输入数据的分布。在 GPT2 模型中，MLP 模块用于对每个 Block 的输出进行特征提取和表示。
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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

    def forward(self, idx):
        """
        前向传播函数
        """
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        """
        Loads a pre-trained GPT-2 model weights from the Hugging Face
        """
        assert model_name in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }
        from transformers import GPT2LMHeadModel

        # n_layer, n_head, n_embd are determined by the model_name
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_name]
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257
        config = GPTConfig(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
