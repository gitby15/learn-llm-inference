import asyncio
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache

from learn_llm_inference.bridge._utils import BaseWorker
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import (
    DecodeRequest,
    RequestContext,
    TokenEvent,
)
from learn_llm_inference.llm_engine.generate import Generator
from learn_llm_inference.llm_engine.sample import sample_token
from learn_llm_inference.llm_engine.kv_cache import split_kv_cache, merge_kv_caches

logger = logging.getLogger(__name__)


class _DecodeWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self._input_queue = GlobalState.get_decode_queue()
        self._decoder = Generator()
        self._eos_token_ids = self._decoder.get_eos_token_ids()
        self._active_contexts: dict[str, RequestContext] = {}

    async def run_forever(self) -> None:
        max_batch = GlobalState.get_max_batch()

        while True:
            batch_requests = [await self._input_queue.get()]
            while len(batch_requests) < max_batch and not self._input_queue.empty():
                batch_requests.append(await self._input_queue.get())
            batch_requests = [
                request
                for request in batch_requests
                if not request.context.cancelled.is_set()
            ]
            if not batch_requests:
                continue
            self._active_contexts = {
                request.context.meta_info.id: request.context
                for request in batch_requests
            }

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
            batch_kv_cache = merge_kv_caches(batch_requests)
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
                try:
                    await self._process_request(
                        request,
                        batch_index,
                        batch_attention_mask,
                        batch_output_logits,
                        updated_batch_cache,
                    )
                except Exception:
                    logger.exception(
                        "Decode request %s failed",
                        request.context.meta_info.id,
                    )
                    await request.context.fail()
                finally:
                    self._active_contexts.pop(
                        request.context.meta_info.id, None
                    )

            # Queue operations may complete immediately; yield so cancellation is observed.
            await asyncio.sleep(0)

    async def _process_request(
        self,
        request: DecodeRequest,
        batch_index: int,
        batch_attention_mask: torch.Tensor,
        batch_output_logits: torch.Tensor,
        updated_batch_cache: DynamicCache,
    ) -> None:
        next_token_id = sample_token(
            batch_output_logits[batch_index, -1, :]
        ).view(1, 1)
        generated_len = request.generated_len + 1
        token_id = int(next_token_id.item())
        finish_reason = None

        if generated_len >= request.context.meta_info.max_tokens:
            finish_reason = "length"
        if token_id in self._eos_token_ids:
            finish_reason = "stop"

        emitted = await request.context.emit(
            TokenEvent(
                token_id=token_id,
                generated_len=generated_len,
                finish_reason=finish_reason,
            )
        )
        if not emitted or finish_reason is not None:
            return

        valid_mask = batch_attention_mask[batch_index].bool()
        request_cache = split_kv_cache(
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
                context=request.context,
                token_id=next_token_id,
                attention_mask=request_attention_mask,
                kv_cache=request_cache,
                generated_len=generated_len,
            )
        )

    async def handle_error(self, error: Exception) -> None:
        failed_contexts = self._active_contexts
        self._active_contexts = {}
        for request_id, context in failed_contexts.items():
            logger.error("Decode request %s failed: %s", request_id, error)
            await context.fail()
