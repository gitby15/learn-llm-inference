import asyncio

import torch
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import DetokenizeRequest
from learn_llm_inference.scheduler.generate import Generator
from learn_llm_inference.scheduler.tokenize import Tokenizer


class _DecodeWorker:
    def __init__(self):
        self._input_queue = GlobalState.get_decode_queue()
        self._detokenize_map = GlobalState.get_detokenize_map()
        self._task = asyncio.create_task(self._process_task(), name="decode-worker")
        self._decoder = Generator()
        self._padding_id = Tokenizer().get_padding_id()
        self._eos_id = Tokenizer().get_eos_id()

    async def _process_task(self) -> None:
        max_batch = GlobalState.get_max_batch()
        batch_decode_req = []

        while True:
            batch_input_ids = []
            batch_attention_mask = []
            batch_kv_cache = []
            while len(batch_decode_req) < max_batch or self._input_queue.empty():
                decode_req = await self._input_queue.get()
                batch_input_ids.append(decode_req.input_ids)
                batch_attention_mask.append(decode_req.attention_mask)
                batch_kv_cache.append(decode_req.kv_cache)
                batch_decode_req.append(decode_req)

            # 补齐长度
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_mask = torch.stack(batch_attention_mask)
            batch_kv_cache = torch.stack(batch_kv_cache)

            decode_result = self._decoder.decode(
                batch_input_ids, batch_attention_mask, batch_kv_cache
            )

            for result, decode_req in zip(decode_result, batch_decode_req):
                id = decode_req.id
                detokenize_queue = self._detokenize_map.get(id, None)
                if detokenize_queue is None:
                    detokenize_queue = asyncio.Queue()
                    self._detokenize_map[id] = detokenize_queue

                detokenize_req = DetokenizeRequest(
                    id=id,
                    logits=result.logits,
                )
                await detokenize_queue.put(detokenize_req)

decode_worker = _DecodeWorker()