"""
TinyShakespeare 字符级预处理
============================
下载 Karpathy 用的 TinyShakespeare 原始文本，把"每个字符"当作一个 token
（不做 BPE 分词，词表 = 文本中出现过的全部字符，vocab≈65）。

输出：train.bin / val.bin（uint16 的字符 id 序列）+ meta.pkl（词表映射）

运行：uv run python data/shakespeare_char/prepare.py
"""

import os
import pickle
import urllib.request

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download():
    """input.txt 不存在时下载 TinyShakespeare。"""
    if os.path.exists(INPUT_FILE):
        return
    urllib.request.urlretrieve(DATA_URL, INPUT_FILE)


def main():
    download()
    with open(INPUT_FILE, encoding="utf-8") as f:
        text = f.read()
    print(f"字符总数: {len(text):,}")

    # 词表 = 文本中出现过的全部字符（每个字符是一个 token）
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id
    itos = {i: c for i, c in enumerate(chars)}  # id -> 字符
    print(f"词表大小: {len(chars)}")

    # 9:1 切分 train / val 并编码为 id 序列
    n = len(text)
    train_ids = [stoi[c] for c in text[: int(n * 0.9)]]
    val_ids = [stoi[c] for c in text[int(n * 0.9) :]]

    # 存 .bin（uint16）+ 词表信息
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "itos": itos, "stoi": stoi}, f)
    print(f"train {len(train_ids):,} token | val {len(val_ids):,} token，预处理完成")


if __name__ == "__main__":
    main()
