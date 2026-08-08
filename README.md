# GPT-2 from Scratch

从零实现 GPT-2（Decoder-only Transformer）语言模型的项目，以 GPT-2 base（124M 参数）为对标目标，用作学习 LLM 预训练与后训练全流程的练手工程。预训练沿用 nanoGPT（Karpathy）的做法，直接用 TinyShakespeare 预训练，分词采用 **GPT-2 BPE**（tiktoken，vocab 50257）——现代大模型标准的子词分词方式。

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
- **因果掩码**：注意力使用因果掩码保证自回归性质（每个位置只能看到自身及之前的位置）；
- **标准初始化**：所有 Linear / Embedding 用 N(0, 0.02)，残差投影 c_proj 额外缩放 0.02/√(2·n_layer)（GPT-2 论文做法），防止深层残差流累积变大。

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
├── data/              # 数据集（TinyShakespeare 预训练语料）
│   └── shakespeare/       # BPE 预处理（GPT-2 BPE 分词，vocab 50257）
├── model/             # 训练产出的模型 checkpoint（已 gitignore）
├── docs/              # 技术文档（from_pretrained 权重加载原理与踩坑记录）
├── src/               # 核心源码
│   ├── config.py      # GPTConfig / TrainConfig 超参数配置（dataclass）
│   ├── model.py       # GPT2 模型定义（注意力 / MLP / Block / GPT2）
│   ├── dataset.py     # BPE 数据集：加载与 batch 采样
│   ├── pretrain.py    # 预训练（TinyShakespeare，参考 nanoGPT）
│   ├── finetune.py    # 后训练 / 监督微调（SFT）流程
│   └── generate.py    # 文本生成（贪心 / Top-k / Nucleus 采样）
├── main.py            # 程序入口（自定义启动各程序，当前启动预训练）
├── pyproject.toml     # 依赖与项目元信息（uv 管理）
└── README.md
```

## 环境准备

- Python ≥ 3.13（推荐使用 [uv](https://docs.astral.sh/uv/) 管理）
- 依赖：`torch`、`torchinfo`、`transformers`、`tiktoken`、`numpy`

```bash
uv sync
```

## 加载官方权重推理

在 `main.py` 中调用 `use_hf_pretrained_weights()` 可加载 Hugging Face 官方预训练权重并生成文本（`view_model_structure()` 可单独查看模型结构与参数量）：

```bash
uv run python main.py
```

首次运行会自动下载权重（缓存到 HF 缓存目录）。也可以直接用代码加载：

```python
from src.model import GPT2

model = GPT2.from_pretrained("gpt2")  # 或 gpt2-medium / gpt2-large / gpt2-xl
```

> 注：OpenAI 官方 checkpoint 中的注意力与 MLP 投影层以 `Conv1D` 存储，导入到标准 `nn.Linear` 时需要转置权重，这一转换已在 `GPT2.from_pretrained` 中处理；Hugging Face 独有的 `attn.bias` / `attn.masked_bias` 缓冲区会被自动过滤。

> 注：关于 `from_pretrained` 的完整加载原理（Conv1D 转置、缓冲区过滤）以及一个因后缀匹配误伤投影偏置的 bug 与修复过程，见 [docs/from_pretrained.md](docs/from_pretrained.md)。

## 预训练

直接用 TinyShakespeare 预训练，分词采用 GPT-2 BPE（tiktoken，vocab 50257），模型保持 GPT-2 124M 架构（即 `config.py` 默认值：12 层 / 12 头 / 768 维 / block 1024）。

```bash
# 1. 准备数据（下载 TinyShakespeare → BPE 分词 → train.bin / val.bin + meta.pkl）
uv run python data/shakespeare/prepare.py

# 2. 预训练（由 main.py 启动，超参数在 src/config.py 中设置）
uv run python main.py
```

| 超参数 | 值 |
| --- | --- |
| tokenizer | GPT-2 BPE（tiktoken） |
| vocab_size | 50257 |
| 模型 | n_layer=12, n_head=12, n_embd=768（124M，config.py 默认） |
| block_size / batch_size | 1024 / 4 |
| 优化器 | AdamW（权重衰减 0.1，仅作用于 2D 权重） |
| 学习率 | 3e-4（常量） |
| 训练步数 | 1000 |
| 评估 / 采样 | eval_interval=500、eval_iters=10、sample_every=200 |

训练时每 `eval_interval` 步在 train / val 上评估 loss，val loss 创新低就把 checkpoint 存到 `model/shakespeare/ckpt.pt`；每 `sample_every` 步打印一段续写文本，直观看到模型学会说人话的过程。想快速验证，改 `config.py` 中 `TrainConfig` 的超参数即可。

> 说明：124M 参数对 TinyShakespeare（BPE 后仅 ~32 万 token）明显过参数化，且 CPU 训练很慢，建议在有 GPU 的环境跑，或先调小 `batch_size` / `max_iters` 体会流程。
>
> 说明：预训练与 `from_pretrained` 加载 OpenAI 官方 124M 权重推理（见上文）共用同一套模型代码，只是前者用 TinyShakespeare 从零训练、后者载入官方权重。
>
> 说明：nanoGPT 原版还用了线性 warmup + 余弦退火的学习率调度，本项目为便于学习先用常量学习率；掌握训练主流程后，可参考 nanoGPT 的 `get_lr` 自行加上。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `src/config.py` | 定义 `GPTConfig`（模型架构：`block_size`、`vocab_size`（50257）、`n_layer`、`n_head`、`n_embd`）与 `TrainConfig`（训练超参数：`batch_size`、`learning_rate`、`max_iters` 等） |
| `src/model.py` | 模型实现：`CausalSelfAttention`（因果多头自注意力）、`MLP`（GELU）、`Block`（残差 + Pre-norm）、`GPT2`（完整模型，含 `from_pretrained` 类方法） |
| `src/dataset.py` | TinyShakespeare BPE 数据集：按需加载 train.bin / val.bin（memmap），`get_batch()` 随机采样 batch，`decode()` 用 tiktoken 解码文本 |
| `src/pretrain.py` | 预训练流程（参考 nanoGPT）：模型架构取 `config.py` 的 `GPTConfig` 默认值，训练超参数取自 `TrainConfig`，AdamW（2D 权重衰减）、定期评估 train/val loss 并保存最优 checkpoint、采样文本看进度 |
| `src/finetune.py` | 后训练 / 监督微调（SFT）流程 |
| `src/generate.py` | 交互式文本生成，支持 Greedy、Top-k、Nucleus（top-p）、Temperature 采样 |

## 参考

- [OpenAI GPT-2](https://openai.com/research/better-language-models/)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT
