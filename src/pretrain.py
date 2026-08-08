"""
TinyShakespeare 预训练（GPT-2 BPE，参考 nanoGPT）
================================================
用 TinyShakespeare 语料预训练 GPT-2 124M，分词采用 GPT-2 BPE
（tiktoken，vocab 50257，现代大模型标准的子词分词）。

模型架构超参数在 config.py 的 GPTConfig，训练超参数在 config.py 的 TrainConfig，
本文件只负责训练流程：加载数据 → 构建模型 → AdamW 训练 → 定期评估 / 采样。

运行：由 main.py 启动
"""

import os
import time

import torch
import torch.nn.functional as F

from src.config import GPTConfig, TrainConfig
from src.dataset import decode, get_batch, load_meta
from src.model import GPT2

train_cfg = TrainConfig()  # 训练超参数（config.py）
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model, config):
    """在 train / val 上各取 eval_iters 个 batch 的平均 loss。"""
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(train_cfg.eval_iters):
            x, y = get_batch(split, config.block_size, train_cfg.batch_size, device)
            logits = model(x)
            total += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ).item()
        losses[split] = total / train_cfg.eval_iters
    model.train()
    return losses


@torch.no_grad()
def sample(model, config, max_new_tokens=100, temperature=0.8):
    """随机取一段验证集文本当开头，让模型续写，直观展示学习进度。"""
    x, _ = get_batch("val", config.block_size, 1, device)
    for _ in range(max_new_tokens):
        logits = model(x[:, -config.block_size :])[:, -1, :] / temperature
        x = torch.cat((x, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
    print("-" * 60)
    print(decode(x[0].tolist()))
    print("-" * 60)


def main():
    # 1. 数据与模型：架构取 config.py 的 GPTConfig 默认值（12 层 / 12 头 / 768 维），词表大小从 meta.pkl 推断（BPE 分词产物，50257）
    vocab_size = load_meta()["vocab_size"]
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"vocab_size={config.vocab_size} | block_size={config.block_size} "
        f"| 参数量 {n_params:.1f}M"
    )

    # 2. 优化器：只对 2D 权重（embedding / matmul）做权重衰减，偏置和 LayerNorm 权重不衰减（nanoGPT 的做法）。lm_head 与 wte 是共享张量，named_parameters 会重复产出，需按 id 去重
    decay, nodecay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else nodecay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
    )

    # 3. 训练循环
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for step in range(train_cfg.max_iters):
        x, y = get_batch("train", config.block_size, train_cfg.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"step {step:5d} | loss {loss.item():.4f} | 用时 {time.time()-t0:.0f}s"
            )

        # 定期评估，val loss 创新低就把 checkpoint 存盘
        if step % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, config)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    model.state_dict(), os.path.join(train_cfg.out_dir, "ckpt.pt")
                )
                print(f"已保存最优 checkpoint -> {train_cfg.out_dir}/ckpt.pt")

        # 定期采样文本看进度
        if step % train_cfg.sample_every == 0:
            sample(model, config)

    print(f"训练完成，最优 val loss = {best_val:.4f}")
