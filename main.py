import torch
import torch.nn.functional as F
from torchinfo import summary

from src.model import GPT2
from src.config import GPTConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")


def view_model_structure():
    """
    查看模型结构和参数量。
    """

    config = GPTConfig()
    gpt2 = GPT2(config)
    x = torch.randint(0, config.vocab_size, (1, config.block_size))
    print(x.shape)
    summary(gpt2, input_data=x)


def use_hf_pretrained_weights():
    """
    使用 Hugging Face 官方预训练权重来进行推理。
    """

    num_return_sequences = 5
    max_length = 30

    model = GPT2.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")  # 这将被编码为8个token
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)  # x is (B, T) = (5, 8)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            x = torch.cat((x, xcol), dim=1)
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    # use_hf_pretrained_weights()

    from src.pretrain import main

    main()
