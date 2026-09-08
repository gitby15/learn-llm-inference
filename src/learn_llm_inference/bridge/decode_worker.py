import asyncio
from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache

from learn_llm_inference.bridge._utils import worker_lifecycle
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.bridge.prefill_worker import _split_kv_cache
from learn_llm_inference.data_model import DecodeRequest, ResponseRequest
from learn_llm_inference.llm_engine.generate import Generator
from learn_llm_inference.llm_engine.sample import sample_token


def _merge_kv_caches(requests: list[DecodeRequest]) -> DynamicCache:
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


@worker_lifecycle
class _DecodeWorker:
    def __init__(self):
        self._input_queue = GlobalState.get_decode_queue()
        self._decoder = Generator()
        self._eos_token_ids = self._decoder.get_eos_token_ids()

    async def _process_task(self) -> None:
        max_batch = GlobalState.get_max_batch()

        while True:
            first_request = await self._input_queue.get()
            batch_requests = [first_request]
            while len(batch_requests) < max_batch and not self._input_queue.empty():
                batch_requests.append(await self._input_queue.get())

            batch_input_ids = torch.cat(
                [request.token_id.reshape(1, -1) for request in batch_requests],
                dim=0,
            )
            batch_attention_mask = pad_sequence(
                [request.attention_mask.reshape(-1) for request in batch_requests],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )
            batch_kv_cache = _merge_kv_caches(batch_requests)
            input_length = batch_input_ids.shape[-1]
            position_ids = (
                batch_attention_mask.long().cumsum(dim=-1)[:, -input_length:] - 1
            ).clamp_min(0)

            decode_result = self._decoder.decode(
                batch_input_ids,
                batch_attention_mask,
                batch_kv_cache,
                position_ids=position_ids,
            )
            batch_output_logits = decode_result.logits
            updated_batch_cache = decode_result.past_key_values

            if batch_output_logits is None:
                raise RuntimeError("decode logits is None")
            if not isinstance(updated_batch_cache, DynamicCache):
                raise TypeError(
                    f"Expected DynamicCache, got {type(updated_batch_cache).__name__}"
                )

            for batch_index, request in enumerate(batch_requests):
                next_token_id = sample_token(
                    batch_output_logits[batch_index, -1, :]
                ).view(1, 1)
                generated_len = request.generated_len + 1
                token_id = int(next_token_id.item())
                finish_reason = None
                if token_id in self._eos_token_ids:
                    finish_reason = "stop"

                response_queue = GlobalState.get_response_queue(request.id)
                await response_queue.put(
                    ResponseRequest(
                        id=request.id,
                        token_id=token_id,
                        generated_len=generated_len,
                        finished=finish_reason is not None,
                        finish_reason=finish_reason,
                    )
                )
                if finish_reason is not None:
                    continue

                valid_mask = batch_attention_mask[batch_index].bool()
                request_cache = _split_kv_cache(
                    updated_batch_cache,
                    batch_index,
                    valid_mask,
                )
                request_attention_mask = batch_attention_mask[batch_index, valid_mask]
                request_attention_mask = torch.cat(
                    (
                        request_attention_mask,
                        request_attention_mask.new_ones(1),
                    )
                ).unsqueeze(0)

                await self._input_queue.put(
                    DecodeRequest(
                        id=request.id,
                        token_id=next_token_id,
                        attention_mask=request_attention_mask,
                        kv_cache=request_cache,
                        generated_len=generated_len,
                    )
                )

            # Queue operations may complete immediately; yield so cancellation is observed.
            await asyncio.sleep(0)


if __name__ == "__main__":
    from learn_llm_inference.bridge.bridge_engine import bridge_instance
    from learn_llm_inference.data_model import TokenizeRequest

    async def test():
        await bridge_instance.start()
        try:
            request = TokenizeRequest(
                id="123",
                messages=[
                    {
                        "role": "user",
                        "content": "Who are you? Please briefly introduce yourself.",
                    }
                ],
            )

            response_worker = await bridge_instance.commit_request(request)
            response = await anext(response_worker.get_stream_response())

        finally:
            await bridge_instance.stop()

    asyncio.run(test(), debug=True)
