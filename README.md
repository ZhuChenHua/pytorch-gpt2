# GPT-2 from Scratch

从零实现 GPT-2（Decoder-only Transformer）语言模型的项目，以 GPT-2 base（124M 参数）为对标目标，用作学习 LLM 预训练与后训练全流程的练手工程。

## 模型架构

GPT-2 风格 Decoder-only Transformer：

```
Token Embedding + Positional Embedding
→ N × Block:
      Pre-LayerNorm → Causal Multi-Head Self-Attention → 残差连接
      Pre-LayerNorm → MLP (GELU)                       → 残差连接
→ Final LayerNorm
→ LM Head（与 Token Embedding 权重共享）
```

核心设计：

- **Pre-norm**：在每个子层前做 LayerNorm，训练更稳定；
- **权重共享**：token embedding 与 LM head 共享参数，减少参数量；
- **因果掩码**：注意力使用因果掩码保证自回归性质（每个位置只能看到自身及之前的位置）。

支持从 Hugging Face 官方仓库加载四种规模的预训练权重：

| 模型 | 层数 n_layer | 头数 n_head | 维度 n_embd | 参数量 |
| --- | --- | --- | --- | --- |
| `gpt2` | 12 | 12 | 768 | 124M |
| `gpt2-medium` | 24 | 16 | 1024 | 350M |
| `gpt2-large` | 36 | 20 | 1280 | 774M |
| `gpt2-xl` | 48 | 25 | 1600 | 1558M |

## 项目结构

```
gpt2_124m/
├── data/              # 数据集（如 WikiText-2 等预训练语料）
├── tokenizer/         # 加载的 GPT-2 BPE tokenizer 参数
├── model/             # 训练产出的模型 checkpoint
├── src/               # 核心源码
│   ├── config.py      # GPTConfig 超参数配置（dataclass）
│   ├── model.py       # GPT2 模型定义（注意力 / MLP / Block / GPT2）
│   ├── dataset.py     # 数据集加载与预处理（WikiText-2 Causal LM）
│   ├── pretrain.py    # 预训练（Pretrain）流程
│   ├── finetune.py    # 后训练 / 监督微调（SFT）流程
│   └── generate.py    # 文本生成（贪心 / Top-k / Nucleus 采样）
├── main.py            # 程序入口（模型构建与测试）
├── pyproject.toml     # 依赖与项目元信息（uv 管理）
└── README.md
```

## 环境准备

- Python ≥ 3.13（推荐使用 [uv](https://docs.astral.sh/uv/) 管理）
- 依赖：`torch`、`torchinfo`、`transformers`、`numpy`

```bash
uv sync
```

## 运行

入口程序为 `main.py`，当前用于构建模型并打印模型结构与参数量：

```bash
uv run python main.py
```

加载 Hugging Face 预训练权重（首次运行会联网下载权重，缓存到 HF 缓存目录）：

```python
from src.model import GPT2

model = GPT2.from_pretrained("gpt2")  # 或 gpt2-medium / gpt2-large / gpt2-xl
```

> 注：OpenAI 官方 checkpoint 中的注意力与 MLP 投影层以 `Conv1D` 存储，导入到标准 `nn.Linear` 时需要转置权重，这一转换已在 `GPT2.from_pretrained` 中处理；Hugging Face 独有的 `attn.bias` / `attn.masked_bias` 缓冲区会被自动过滤。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `src/config.py` | 定义 `GPTConfig`：`block_size`（最大序列长度）、`vocab_size`（50257）、`n_layer`、`n_head`、`n_embd` |
| `src/model.py` | 模型实现：`CausalSelfAttention`（因果多头自注意力）、`MLP`（GELU）、`Block`（残差 + Pre-norm）、`GPT2`（完整模型，含 `from_pretrained` 类方法） |
| `src/dataset.py` | 通过 `datasets` 下载 WikiText-2，用 GPT-2 BPE tokenizer 编码后拼接为长序列，按 `seq_len` 切分为不重叠滑动窗口；目标序列为输入序列左移一位（next-token prediction） |
| `src/pretrain.py` | 预训练流程：数据加载 → 前向/反向 → 梯度更新与日志记录 |
| `src/finetune.py` | 后训练 / 监督微调（SFT）流程 |
| `src/generate.py` | 交互式文本生成，支持 Greedy、Top-k、Nucleus（top-p）、Temperature 采样 |

## 参考

- [OpenAI GPT-2](https://openai.com/research/better-language-models/)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT
