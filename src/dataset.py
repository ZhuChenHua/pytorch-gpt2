"""
TinyShakespeare 数据集（GPT-2 BPE 分词）
=======================================
参考 nanoGPT（Karpathy）train.py 的 "poor man's data loader"。

数据用 GPT-2 BPE tokenizer（tiktoken）分词，vocab 50257，这是现代大模型
标准的子词分词方式。由 data/shakespeare/prepare.py 预处理，产物是
train.bin / val.bin（uint16 的 token id 序列）+ meta.pkl。

本模块负责按需加载（memmap）并随机采样 batch：
输入 x 与目标 y 一一对应，y 是 x 右移一位（next-token prediction）。
"""

import os
import pickle

import numpy as np
import torch
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "shakespeare")
ENC = tiktoken.get_encoding("gpt2")


def _require_data():
    """检查数据集是否已由 data/shakespeare/prepare.py 预处理，缺失则给出明确报错。"""
    missing = [
        f
        for f in ("train.bin", "val.bin", "meta.pkl")
        if not os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"数据集未预处理：data/shakespeare/ 缺少 {', '.join(missing)}，"
            "请先运行 uv run python data/shakespeare/prepare.py"
        )


def load_meta():
    """读取 meta.pkl，返回 {vocab_size, tokenizer}。"""
    _require_data()
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        return pickle.load(f)


def decode(ids):
    """把 BPE token id 序列解码回字符串（tiktoken gpt2 词表）。"""
    return ENC.decode([int(i) for i in ids])


def get_batch(split, block_size, batch_size, device):
    """
    从 train / val 中随机采样一个 batch。

    x: (batch_size, block_size) 输入 token 序列
    y: (batch_size, block_size) 目标 token 序列（x 右移一位）
    """
    _require_data()
    # 每次调用重建 memmap，避免长期持有导致内存泄漏（nanoGPT 的做法）
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)
