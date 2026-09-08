from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        if self.top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")


# 输入形状：[vocab_size]
# 输出形状：零维标量，只包含一个值
def sample_token(
    logits: torch.Tensor,
    params: SamplingParams | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one token from a one-dimensional vocabulary logits tensor."""
    if logits.ndim != 1:
        raise ValueError(f"expected one-dimensional logits, got shape {logits.shape}")

    params = params or SamplingParams()
    if params.temperature == 0:
        return logits.argmax(dim=-1)

    filtered_logits = logits / params.temperature

    if 0 < params.top_k < filtered_logits.numel():
        threshold = torch.topk(filtered_logits, params.top_k).values[-1]
        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < threshold,
            float("-inf"),
        )

    if params.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits,
            descending=True,
        )
        cumulative_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        ).cumsum(dim=-1)
        tokens_to_remove = cumulative_probabilities > params.top_p
        tokens_to_remove[1:] = tokens_to_remove[:-1].clone()
        tokens_to_remove[0] = False
        remove_mask = torch.zeros_like(tokens_to_remove).scatter(
            0,
            sorted_indices,
            tokens_to_remove,
        )
        filtered_logits = filtered_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

    probabilities = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(0)
