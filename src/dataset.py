"""
GPT 数据处理 — WikiText-2 因果语言模型（Causal LM）数据集
==========================================================
处理流程：
  1. 通过 HuggingFace `datasets` 下载 WikiText-2
  2. 使用 GPT-2 BPE tokenizer（词表大小 50 257）编码全文
  3. 将所有 token 拼接为一个长序列
  4. 按 `seq_len` 切分为不重叠的滑动窗口
  5. 目标序列 = 输入序列左移 1 位（next-token prediction）
"""
