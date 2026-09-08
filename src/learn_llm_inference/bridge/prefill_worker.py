import asyncio
from typing import cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache

from learn_llm_inference.bridge._utils import Worker, worker_lifecycle
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import DecodeRequest, ResponseRequest
from learn_llm_inference.llm_engine.generate import Generator
from learn_llm_inference.llm_engine.sample import sample_token
from learn_llm_inference.llm_engine.tokenize import Tokenizer
from learn_llm_inference.llm_engine.kv_cache import split_kv_cache




@worker_lifecycle
class _PrefillWorker:
    def __init__(self):
        self._prefill_queue = GlobalState.get_prefill_queue()
        self._next_queue = GlobalState.get_decode_queue()
        self._prefiller = Generator()
        self._eos_token_ids = self._prefiller.get_eos_token_ids()
        self._padding_id = Tokenizer().get_padding_id()

    async def _process_task(self) -> None:
        _max_batch = GlobalState.get_max_batch()
        while True:
            # 等待第一个req，避免陷入无效循环
            batch_requests = [await self._prefill_queue.get()]
            # 收集一个batch的req
            while len(batch_requests) < _max_batch and not self._prefill_queue.empty():
                batch_requests.append(self._prefill_queue.get_nowait())

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
            prefill_result = self._prefiller.prefill(batch_input_ids, batch_attention_mask)

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
                bool_mask = batch_attention_mask[batch_index].bool()
                request_logits = batch_output_logits[batch_index, bool_mask, :]
                next_token_id = sample_token(request_logits[-1]).view(1, 1)
                # 因为多生成了一个token，所以原始mask需要多加一位，值为1（因为不需要被ignore）
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

                response_req = ResponseRequest(
                    id=task_req.id,
                    token_id=token_id,
                    generated_len=1,
                    finished=finish_reason is not None,
                    finish_reason=finish_reason,
                )
                response_queue = GlobalState.get_response_queue(task_req.id)
                await response_queue.put(response_req)

                # 如果在prefill的时候就结束了，就不用走下一个阶段了
                if finish_reason is not None:
                    continue

                decode_req = DecodeRequest(
                    id=task_req.id,
                    token_id=next_token_id,
                    attention_mask=request_attention_mask,
                    kv_cache=request_kv_cache,
                    generated_len=1,
                )
                await self._next_queue.put(decode_req)


if __name__ == "__main__":
    from learn_llm_inference.bridge.bridge_engine import bridge_instance
    from learn_llm_inference.bridge.tokenize_worker import _TokenizeWorker
    from learn_llm_inference.data_model import TokenizeRequest


    async def test():
        workers = (
            cast(Worker, _TokenizeWorker()),
            cast(Worker, _PrefillWorker()),
        )
        for worker in workers:
            worker.start()
        try:
            request = TokenizeRequest(
                id="123",
                messages=[{"role": "user", "content": "Who are you? Please briefly introduce yourself."}],
            )

            await bridge_instance.commit_request(request)
            decode_request = await GlobalState.get_decode_queue().get()

        finally:
            await asyncio.gather(*(worker.stop() for worker in workers))

    asyncio.run(test(), debug=True)
