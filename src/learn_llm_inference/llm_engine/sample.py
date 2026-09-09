# Todo：现在是贪心解码，未来补充其他采样策略，如温度和TopK等

import torch

# 输入形状：[vocab_size]
# 输出形状：零维标量，只包含一个值
def sample_token(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Select one token from a one-dimensional vocabulary logits tensor."""
    if logits.ndim != 1:
        raise ValueError(f"expected one-dimensional logits, got shape {logits.shape}")
    # 贪心解码，选择概率最大的那个token
    return logits.argmax(dim=-1)

