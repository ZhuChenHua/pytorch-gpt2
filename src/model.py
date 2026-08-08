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
  - 因果掩码：注册一个下三角 mask buffer（self.bias），
              在 softmax 前把上三角掩为 -inf，保证自回归性质
"""

import math

import torch
import torch.nn as nn

from src.config import GPTConfig, cache_dir


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention 模块：
    将输入线性投影为 query、key、value 三个矩阵 → 计算缩放点积注意力
    → 施加因果掩码（每个位置只能关注自身及之前的位置，保证自回归性质）
    → 对 value 加权求和 → 输出线性投影（c_proj）。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # query, key, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # attention hyperparameters
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
        # project x -> (B, T, 3*C)
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
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    MLP 模块：
    对输入做线性变换（c_fc，特征维度放大 4 倍）→ GELU 激活 → 线性变换（c_proj，投影回原维度），
    为模型引入非线性并扩大表示容量。GELU 采用 tanh 近似（与 OpenAI 官方实现一致），
    是平滑可微的 ReLU 变体，能在保持非线性的同时缓解神经元"死亡"问题。
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

        # 权重共享：lm_head 与 token embedding 绑定同一份权重（减少参数量，124M）。
        # 注意方向：让 wte 沿用 lm_head 的 nn.Linear 小初始化，而不是反过来——
        # 若 lm_head 沿用 nn.Embedding 的 N(0,1) 初始化，初始 logits 会过大，loss 爆掉。
        self.transformer.wte.weight = self.lm_head.weight

        # 权重初始化（GPT-2 / nanoGPT 标准做法）：
        #   - 所有 Linear / Embedding 统一用 N(0, 0.02)（而非 PyTorch 默认的 kaiming / N(0,1)）。
        #     默认初始化下位置嵌入 wpe 是 N(0,1)、std≈1，而共享后的 token 嵌入 wte std≈0.02，
        #     位置信号比 token 信号大近 50 倍，初始表示几乎被位置主导。
        #   - 残差投影 c_proj 额外用 0.02/sqrt(2*n_layer) 缩放（GPT-2 论文），
        #     防止残差流在深层逐层累积变大。
        # 此时 wte 与 lm_head 共享同一份权重，apply 会遍历两处各初始化一次，分布一致，无冲突。
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        前向传播：
        输入 token 序列 idx（形状 (B, T)），
        返回各位置对所有 token 的 logits（形状 (B, T, vocab_size)）。
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
        # 通过广播机制将位置嵌入加到 token 嵌入上，得到形状为 (B, T, n_embd) 的输入表示
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
        从 Hugging Face Hub 加载 GPT-2 官方预训练权重，
        返回一个参数已对齐（含 Conv1D 转置处理）的新 GPT2 模型。
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

        # 加载 Hugging Face 官方预训练权重，缓存目录设置为项目根目录下的 cache 文件夹
        hf_model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        sd_hf = hf_model.state_dict()

        # Hugging Face 模型额外带有 attn.bias / attn.masked_bias 两个缓冲区（因果掩码），
        # 本实现使用自注册的下三角 mask buffer（self.bias），不需要它们，直接过滤掉。
        # 注意：后缀必须带前导点（".attn.bias"），否则会误伤
        # attn.c_attn.bias / attn.c_proj.bias 两个投影偏置，导致它们保留随机初始化值。
        keys = [
            k
            for k in sd_hf
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        # Hugging Face 官方 checkpoint 中这些投影层以 Conv1D 存储（transformers v5 中权重形状为 [in, out]），
        # 而本地实现使用标准 nn.Linear（权重形状为 [out, in]），导入时需要转置
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        sd = model.state_dict()
        # 逐层拷贝权重，并校验名称与形状完全对齐
        with torch.no_grad():
            for k in keys:
                if any(k.endswith(w) for w in transposed):
                    # 特殊处理：Conv1D 权重需要转置
                    assert (
                        sd_hf[k].shape[::-1] == sd[k].shape
                    ), f"shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                    sd[k].copy_(sd_hf[k].t())
                else:
                    # 其余参数直接拷贝
                    assert (
                        sd_hf[k].shape == sd[k].shape
                    ), f"shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                    sd[k].copy_(sd_hf[k])

        return model
