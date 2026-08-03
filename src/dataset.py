"""
TinyShakespeare 数据集（字符级 / BPE 两种）
==========================================
参考 nanoGPT（Karpathy）train.py 的 "poor man's data loader"。

支持两个数据集（都在 data/ 下）：
  - shakespeare_char：字符级。每个字符一个 token，词表 65。
    由 data/shakespeare_char/prepare.py 预处理。
  - shakespeare：GPT-2 BPE 分词，vocab 50257（现代大模型标准的子词分词）。
    由 data/shakespeare/prepare.py 预处理。

预处理产物都是相同的 .bin（uint16 token id 序列）+ meta.pkl。
本模块负责按需加载（memmap）并随机采样 batch：
输入 x 与目标 y 一一对应，y 是 x 右移一位（next-token prediction）。
"""

import os
import pickle
import subprocess
import sys

import numpy as np
import torch
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认数据集（pretrain.py 里通过自己的 DATASET 常量选择）
DATASET = "shakespeare_char"


def _data_dir(dataset):
    return os.path.join(PROJECT_ROOT, "data", dataset)


def ensure_prepared(dataset=DATASET):
    """若该数据集未预处理，则运行对应的 prepare.py。"""
    required = ["train.bin", "val.bin", "meta.pkl"]
    if not all(os.path.exists(os.path.join(_data_dir(dataset), f)) for f in required):
        prepare_py = os.path.join(_data_dir(dataset), "prepare.py")
        print(f"数据集 {dataset} 未准备，正在运行 {prepare_py} ...")
        subprocess.run([sys.executable, prepare_py], check=True)


def load_meta(dataset=DATASET):
    """读取 meta.pkl，返回 {vocab_size, ...}。字符版额外有 itos / stoi。"""
    ensure_prepared(dataset)
    with open(os.path.join(_data_dir(dataset), "meta.pkl"), "rb") as f:
        return pickle.load(f)


def decode(ids, dataset=DATASET):
    """把 token id 序列解码回字符串：字符版用词表映射，BPE 版用 tiktoken。"""
    if dataset == "shakespeare":
        return tiktoken.get_encoding("gpt2").decode([int(i) for i in ids])
    itos = load_meta(dataset)["itos"]
    return "".join(itos[int(i)] for i in ids)


def get_batch(split, block_size, batch_size, device, dataset=DATASET):
    """
    从 train / val 中随机采样一个 batch。

    x: (batch_size, block_size) 输入 token 序列
    y: (batch_size, block_size) 目标 token 序列（x 右移一位）
    """
    ensure_prepared(dataset)
    # 每次调用重建 memmap，避免长期持有导致内存泄漏（nanoGPT 的做法）
    data = np.memmap(
        os.path.join(_data_dir(dataset), f"{split}.bin"), dtype=np.uint16, mode="r"
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)
