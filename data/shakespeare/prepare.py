"""
TinyShakespeare GPT-2 BPE 预处理
================================
下载 TinyShakespeare 原始文本，用 GPT-2 BPE tokenizer 分词
（vocab 50257，现代大模型标准的子词分词；字符版是每个字符一个 token）。

输出：train.bin / val.bin（uint16 的 token id）+ meta.pkl

运行：uv run python data/shakespeare/prepare.py
"""

import os
import pickle
import shutil
import urllib.request

import numpy as np
import tiktoken

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# 字符版已下载过同一份文本，直接复用，避免重复下载
CHAR_INPUT = os.path.join(DATA_DIR, "..", "shakespeare_char", "input.txt")


def ensure_input():
    if os.path.exists(INPUT_FILE):
        return
    if os.path.exists(CHAR_INPUT):
        print(f"复用 {os.path.normpath(CHAR_INPUT)}")
        shutil.copyfile(CHAR_INPUT, INPUT_FILE)
        return
    urllib.request.urlretrieve(DATA_URL, INPUT_FILE)


def main():
    ensure_input()
    with open(INPUT_FILE, encoding="utf-8") as f:
        text = f.read()
    print(f"字符总数: {len(text):,}")

    # GPT-2 BPE 分词（字符版则是 stoi 逐字符映射）
    enc = tiktoken.get_encoding("gpt2")
    n = len(text)
    train_ids = enc.encode(text[: int(n * 0.9)])
    val_ids = enc.encode(text[int(n * 0.9) :])
    print(f"词表大小: {enc.n_vocab}")
    print(f"train {len(train_ids):,} token | val {len(val_ids):,} token")

    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}, f)
    print("预处理完成")


if __name__ == "__main__":
    main()
