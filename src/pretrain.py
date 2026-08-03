"""
TinyShakespeare 预训练（参考 nanoGPT）
======================================
直接用 TinyShakespeare 预训练，支持两个数据集（改顶部的 DATASET 即可切换）：
  - shakespeare_char：字符级，每个字符一个 token（vocab 65），nanoGPT 微型配置
  - shakespeare：GPT-2 BPE 分词（vocab 50257），保持 GPT-2 124M 架构
                  （config.py 默认值：12 层 / 12 头 / 768 维 / block 1024）

流程：加载数据 → 构建 GPT-2 → AdamW 训练 → 定期评估 loss / 采样文本。

运行：uv run python -m src.pretrain
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

# 让 `python src/pretrain.py` 也能直接运行（否则需用 `python -m src.pretrain`）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GPTConfig
from src.dataset import decode, get_batch, load_meta
from src.model import GPT2

# ================ 数据集与超参数 ================
DATASET = "shakespeare_char"  # 或 "shakespeare"（BPE 版，保持 124M 架构）

if DATASET == "shakespeare_char":
    batch_size, block_size = 64, 256
    n_layer, n_head, n_embd = 6, 6, 384
    max_iters, eval_iters = 5000, 100
else:  # BPE 版：语料仅 ~32 万 token 但模型 124M，故 batch 小、步数少
    batch_size, block_size = 4, 1024
    n_layer, n_head, n_embd = 12, 12, 768
    max_iters, eval_iters = 1000, 10

eval_interval = 500   # 每多少步评估一次 train / val loss
sample_every = 1000   # 每多少步打印一段生成的文本
learning_rate = 3e-4
out_dir = os.path.join("model", DATASET)

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model):
    """在 train / val 上各取 eval_iters 个 batch 的平均 loss。"""
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device, dataset=DATASET)
            logits = model(x)
            total += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ).item()
        losses[split] = total / eval_iters
    model.train()
    return losses


@torch.no_grad()
def sample(model, max_new_tokens=300, temperature=0.8):
    """随机取一段验证集文本当开头，让模型续写，直观展示学习进度。"""
    x, _ = get_batch("val", block_size, 1, device, dataset=DATASET)
    for _ in range(max_new_tokens):
        logits = model(x[:, -block_size:])[:, -1, :] / temperature
        x = torch.cat((x, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
    print("-" * 60)
    print(decode(x[0].tolist(), dataset=DATASET))
    print("-" * 60)


def main():
    # 1. 数据与模型（词表大小从 meta.pkl 推断）
    vocab_size = load_meta(DATASET)["vocab_size"]
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"数据集 {DATASET} | vocab_size={vocab_size} | 参数量 {n_params:.1f}M")

    # 2. 优化器：只对 2D 权重（embedding / matmul）做权重衰减，
    #    偏置和 LayerNorm 权重不衰减（nanoGPT 的做法）。
    #    lm_head 与 wte 是共享张量，named_parameters 会重复产出，需按 id 去重
    decay, nodecay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else nodecay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    # 3. 训练循环
    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for step in range(max_iters):
        x, y = get_batch("train", block_size, batch_size, device, dataset=DATASET)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | 用时 {time.time()-t0:.0f}s")

        # 定期评估，val loss 创新低就把 checkpoint 存盘
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(model.state_dict(), os.path.join(out_dir, "ckpt.pt"))
                print(f"已保存最优 checkpoint -> {out_dir}/ckpt.pt")

        # 定期采样文本看进度
        if step % sample_every == 0:
            sample(model)

    print(f"训练完成，最优 val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
