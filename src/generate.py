"""
GPT 文本生成 — 交互式命令行工具
==================================
采样策略：
  - Greedy：       top_k=1，temperature=1.0
  - Top-k：        仅从概率最高的 k 个 token 中采样
  - Nucleus (p)：  从累积概率 ≥ p 的最小 token 集合中采样（top-p / nucleus sampling）
  - Temperature：  在 softmax 之前缩放 logits 的尖锐程度
"""

