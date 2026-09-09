import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache

from learn_llm_inference.bridge._utils import BaseWorker
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import (
    DecodeRequest,
    PrefillRequest,
    RequestContext,
    TokenEvent,
)
from learn_llm_inference.llm_engine.generate import Generator
from learn_llm_inference.llm_engine.sample import sample_token
from learn_llm_inference.llm_engine.tokenize import Tokenizer
from learn_llm_inference.llm_engine.kv_cache import split_kv_cache

logger = logging.getLogger(__name__)


class _PrefillWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self._prefill_queue = GlobalState.get_prefill_queue()
        self._next_queue = GlobalState.get_decode_queue()
        self._prefiller = Generator()
        self._eos_token_ids = self._prefiller.get_eos_token_ids()
        self._padding_id = Tokenizer().get_padding_id()
        self._active_contexts: dict[str, RequestContext] = {}

    async def run_forever(self) -> None:
        _max_batch = GlobalState.get_max_batch()
        while True:
            # 等待第一个req，避免陷入无效循环
            batch_requests = [await self._prefill_queue.get()]
            # 收集一个batch的req
            while len(batch_requests) < _max_batch and not self._prefill_queue.empty():
                request = await self._prefill_queue.get()
                batch_requests.append(request)
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

            # 补齐长度
            batch_input_ids = pad_sequence(
                [request.input_ids.reshape(-1) for request in batch_requests],
                batch_first=True,
                padding_value=self._padding_id,
                padding_side="left",
            )
            batch_attention_mask = pad_sequence(
                [request.attention_mask.reshape(-1) for request in batch_requests],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )

            # prefill这个batch，前面都是CPU任务，prefill是GPU任务
            prefill_result = self._prefiller.prefill(
                batch_input_ids, batch_attention_mask
            )

            # 现在的形状的[B, T, C]
            batch_output_logits = prefill_result.logits
            batch_kv_cache = prefill_result.past_key_values

            # 类型保护
            if batch_output_logits is None:
                raise RuntimeError("prefill logits is None")
            if not isinstance(batch_kv_cache, DynamicCache):
                raise TypeError(
                    f"Expected DynamicCache, got {type(batch_kv_cache).__name__}"
                )

            # 把batch的输出拆成单个req送入下一步
            for batch_index, task_req in enumerate(batch_requests):
                try:
                    await self._process_request(
                        task_req,
                        batch_index,
                        batch_attention_mask,
                        batch_output_logits,
                        batch_kv_cache,
                    )
                except Exception:
                    logger.exception(
                        "Prefill request %s failed",
                        task_req.context.meta_info.id,
                    )
                    await task_req.context.fail()
                finally:
                    self._active_contexts.pop(
                        task_req.context.meta_info.id, None
                    )

    async def _process_request(
        self,
        task_req: PrefillRequest,
        batch_index: int,
        batch_attention_mask: torch.Tensor,
        batch_output_logits: torch.Tensor,
        batch_kv_cache: DynamicCache,
    ) -> None:
        bool_mask = batch_attention_mask[batch_index].bool()
        request_logits = batch_output_logits[batch_index, bool_mask, :]
        next_token_id = sample_token(request_logits[-1]).view(1, 1)
        request_attention_mask = torch.cat(
            (
                batch_attention_mask[batch_index, bool_mask],
                batch_attention_mask.new_ones(1),
            )
        ).unsqueeze(0)

        # Todo(Important): 这一段KV Cache的处理是AI写的，我还没学会
        request_kv_cache = split_kv_cache(
            batch_kv_cache,
            batch_index,
            bool_mask,
        )

        # Todo: 这里会产生一次GPU -> CPU的数据搬运，在循环中会反复多次，后面可以跟sample一起优化成batch处理
        token_id = int(next_token_id.item())
        finish_reason = None
        if token_id in self._eos_token_ids:
            finish_reason = "stop"
        if task_req.context.meta_info.max_tokens <= 1:
            finish_reason = "length"

        emitted = await task_req.context.emit(
            TokenEvent(
                token_id=token_id,
                generated_len=1,
                finish_reason=finish_reason,
            )
        )

        if emitted and finish_reason is None:
            await self._next_queue.put(
                DecodeRequest(
                    context=task_req.context,
                    token_id=next_token_id,
                    attention_mask=request_attention_mask,
                    kv_cache=request_kv_cache,
                    generated_len=1,
                )
            )

    async def handle_error(self, error: Exception) -> None:
        failed_contexts = self._active_contexts
        self._active_contexts = {}
        for request_id, context in failed_contexts.items():
            logger.error(
                "Prefill request %s failed: %s", request_id, error
            )
            await context.fail()
