# 这里实现KV Cache的处理逻辑
from typing import cast

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from learn_llm_inference.data_model import DecodeRequest

def split_kv_cache(
    batch_cache: DynamicCache,
    batch_index: int,
    bool_mask: torch.Tensor,
) -> DynamicCache:
    cache_data: list[tuple[torch.Tensor | None, ...]] = []
    for layer in batch_cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            cache_data.append((None, None))
            continue

        key_mask = bool_mask.to(keys.device)
        value_mask = bool_mask.to(values.device)
        cache_data.append(
            (
                keys[batch_index : batch_index + 1, :, key_mask, :],
                values[batch_index : batch_index + 1, :, value_mask, :],
            )
        )

    return DynamicCache(ddp_cache_data=cache_data)

def merge_kv_caches(requests: list[DecodeRequest]) -> DynamicCache:
    raw_caches = [request.kv_cache for request in requests]
    if not all(isinstance(cache, DynamicCache) for cache in raw_caches):
        cache_types = ", ".join(type(cache).__name__ for cache in raw_caches)
        raise TypeError(f"Expected DynamicCache instances, got: {cache_types}")
    caches = [cast(DynamicCache, cache) for cache in raw_caches]

    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("Cannot batch KV caches with different layer counts")

    cache_data: list[tuple[torch.Tensor | None, ...]] = []
    for layer_index in range(layer_count):
        layer_states: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for cache in caches:
            layer = cache.layers[layer_index]
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            layer_states.append((keys, values))

        if all(keys is None and values is None for keys, values in layer_states):
            cache_data.append((None, None))
            continue
        if any(keys is None or values is None for keys, values in layer_states):
            raise ValueError(f"Inconsistent KV cache state at layer {layer_index}")

        initialized_states = [
            (cast(torch.Tensor, keys), cast(torch.Tensor, values))
            for keys, values in layer_states
        ]
        max_cache_length = max(keys.shape[-2] for keys, _ in initialized_states)
        batched_keys = torch.cat(
            [
                F.pad(keys, (0, 0, max_cache_length - keys.shape[-2], 0))
                for keys, _ in initialized_states
            ],
            dim=0,
        )
        batched_values = torch.cat(
            [
                F.pad(values, (0, 0, max_cache_length - values.shape[-2], 0))
                for _, values in initialized_states
            ],
            dim=0,
        )
        cache_data.append((batched_keys, batched_values))

    return DynamicCache(ddp_cache_data=cache_data)
