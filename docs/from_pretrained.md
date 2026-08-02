# `from_pretrained` 预训练权重加载：原理与踩坑记录

本文记录 `GPT2.from_pretrained`（`src/model.py`）如何把 Hugging Face 官方 GPT-2 权重正确映射到本地实现的模型上，以及加载过程中一个藏得很深的 bug 及其修复过程。

## 一、目的

`from_pretrained` 是一个类方法：从 Hugging Face Hub 下载 GPT-2 官方预训练权重，填入本地 `GPT2` 模型，返回一个参数已对齐的新模型。它是"用自己的代码跑官方权重做推理"的关键入口。

```python
model = GPT2.from_pretrained("gpt2")  # 或 gpt2-medium / gpt2-large / gpt2-xl
```

## 二、背景：两套不同的权重表示

同一个 GPT-2 权重，在两个框架里的存储方式不同——这是"需要映射"的根本原因。

### HF 侧：投影层用 Conv1D

`transformers` 的 GPT-2 把注意力/MLP 的投影层实现为 `Conv1D`（见 `.venv/Lib/site-packages/transformers/pytorch_utils.py`）：

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))  # [in_features, out_features]
        ...
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)  # x @ W + b
        return x.view(size_out)
```

要点：

- **权重形状是 `[in, out]`**（`torch.empty(nx, nf)`），例如 `c_attn` 是 `[768, 2304]`；
- 前向是 `x @ W + b`，**不转置**。

### 本地侧：投影层用标准 nn.Linear

本地 `src/model.py` 用 `nn.Linear` 实现投影层：

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # in=768, out=2304
```

- **权重形状是 `[out, in]`**（`nn.Linear` 的参数约定），例如 `c_attn` 是 `[2304, 768]`；
- 前向是 `x @ W^T + b`。

### 两者如何等价

设 Conv1D 权重为 `W_c`（形状 `[in, out]`）。要让 `nn.Linear` 产生相同结果，只需存 `W_c` 的转置：

```
W_linear = W_c.T                      # [out, in]
x @ W_linear.T + b = x @ (W_c.T).T + b = x @ W_c + b
```

所以导入时**必须转置** `c_attn / c_proj / c_fc / c_proj` 这四个投影层的权重。`wte`、`wpe`、`ln_*`、`lm_head` 的形状与语义在两边完全一致，直接拷贝即可。

## 三、加载流程（对照代码）

`src/model.py` 中 `from_pretrained` 的核心步骤：

```python
# 1. 根据 model_name 确定 n_layer / n_head / n_embd，构造本地模型
config_args = {"gpt2": dict(n_layer=12, n_head=12, n_embd=768), ...}[model_name]
config_args["block_size"] = 1024
config_args["vocab_size"] = 50257
model = GPT2(GPTConfig(**config_args))

# 2. 加载 HF 官方权重
hf_model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
sd_hf = hf_model.state_dict()

# 3. 过滤 HF 特有的掩码缓冲区
keys = [
    k for k in sd_hf
    if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
]

# 4. 需要转置的投影层权重
transposed = ["attn.c_attn.weight", "attn.c_proj.weight",
              "mlp.c_fc.weight",   "mlp.c_proj.weight"]

# 5. 逐 key 拷贝 + shape 校验
sd = model.state_dict()
with torch.no_grad():
    for k in keys:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape  # [in,out] 反转 = [out,in]
            sd[k].copy_(sd_hf[k].t())                    # 转置后拷贝
        else:
            assert sd_hf[k].shape == sd[k].shape
            sd[k].copy_(sd_hf[k])

# 6. 权重共享：lm_head 与 wte 绑定
model.lm_head.weight = model.transformer.wte.weight
```

每一步都有 shape 断言兜底，任何 key 名称或形状对不上都会立刻报错，而不是静默加载出错误权重。

## 四、踩坑记录：`endswith("attn.bias")` 误伤投影偏置

### 现象

加载官方权重后，生成文本严重退化，反复输出重复词：

```
> Hello, I'm a language model, a language model model model model model ...
> Hello, I'm a language model, as an an an an an an an an ...
```

模型架构没问题、大部分权重也正确，但推理结果像是"半随机"。

### 根因

过滤缓冲区的条件写成了：

```python
not k.endswith("attn.bias")
```

本意是去掉 HF 的因果掩码缓冲区 `transformer.h.N.attn.bias`。但 `str.endswith` 是**纯后缀匹配**：

```python
"transformer.h.0.attn.c_attn.bias".endswith("attn.bias")  # True !!!
```

`c_attn.bias`（`[2304]`）和 `c_proj.bias`（`[768]`）这两个投影偏置，也以 `"attn.bias"` 结尾，被这个过滤器一并删掉了。于是它们**从未被拷贝**，一直保留 `GPT2.__init__` 里的随机初始化值。

每层注意力投影的偏置是随机数 → 注意力输出错误 → 越往下越离谱 → 最后 logits 与官方权重完全不同。

### 为什么初次检查没发现

逐 key 对比参数时，检查代码复用了同一个 `endswith("attn.bias")` 来跳过缓冲区的 key，于是 `c_attn.bias` / `c_proj.bias` 在检查时**也被跳过了**，得出"0 个不匹配"的假阴性。检查逻辑与加载逻辑用了同一个有缺陷的条件，错误被互相掩盖。

### 修复

后缀匹配加前导点：

```python
keys = [
    k for k in sd_hf
    if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
]
```

`"c_attn.bias"` 不再匹配 `".attn.bias"`（前导点是 `c` 而不是 `.`）；纯缓冲区 `attn.bias` 仍然匹配。

### 教训

1. `str.endswith` 匹配的是子串后缀，`attn.bias` 会命中 `c_attn.bias`、`c_proj.bias` 等更长的 key。要用带模块边界的完整后缀（前导点 `".attn.bias"`）。
2. 校验逻辑要与加载逻辑保持一致，但不能复用加载里出错的那个条件——否则会得出"一切正常"的错误结论。

## 五、如何验证权重加载正确

把"加载结果"与"官方 HF 模型"做差分，是最可靠的验收方式：

1. **参数级对比**：逐 key 对比本地模型与 HF 模型的 `state_dict()`，处理 Conv1D 转置后，每个 key 的最大绝对差应 < 1e-4；
2. **输出级对比**：给两个模型喂相同输入，对比 logits。修复前最大差 ~149，修复后 ~1e-5（仅浮点噪声），且 top-k token 完全一致；
3. **生成效果**：修复后 `main.py` 生成连贯文本。

修复前后对比：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 参数不匹配数 | 0（假阴性，偏置被跳过） | 0（真实，161 个 key 全对齐） |
| logits 最大绝对差 | ~149 | ~7.6e-5 |
| 最后位置 top-5 token | 与 HF 不同 | 与 HF 完全一致 |
| 生成文本 | 重复词退化成循环 | 连贯有语义 |

## 六、相关文件

- `src/model.py`：`GPT2.from_pretrained` 类方法（核心逻辑与本次修复所在）
- `src/config.py`：`GPTConfig`、`cache_dir`
- `main.py`：`use_hf_pretrained_weights` 推理入口（加载 + 生成）
