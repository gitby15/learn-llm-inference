import asyncio

import torch
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import PrefillRequest, DecodeRequest
from learn_llm_inference.scheduler.generate import Generator
from learn_llm_inference.scheduler.tokenize import Tokenizer


class _PrefillWorker:
    def __init__(self):
        self._prefill_queue = GlobalState.get_prefill_queue()
        self._next_queue = GlobalState.get_decode_queue()
        self._prefiller = Generator()
        self._padding_id = Tokenizer().get_padding_id()
        self._task = asyncio.create_task(self._process_task(), name="prefill-worker")
        pass

    async def _process_task(self) -> None:
        _batch_prefill_request = []
        _max_batch = GlobalState.get_max_batch()
        print("启动prefill")
        while True:
            batch_input_ids = []
            batch_attention_mask = []
            # 凑齐batch
            while len(_batch_prefill_request) < _max_batch and not self._prefill_queue.empty():
                print("加载prefill request: ", len(_batch_prefill_request))
                task_req = await self._prefill_queue.get()
                batch_input_ids.append(task_req.input_ids)
                batch_attention_mask.append(task_req.attention_mask)
                _batch_prefill_request.append(task_req)
            
            print("凑齐Batch成功", len(_batch_prefill_request))
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_mask = torch.stack(batch_attention_mask)
            batch_output_logits, batch_kv_cache = self._prefiller.prefill(batch_input_ids, batch_attention_mask)

            print("prefill成功, kv_cache:", batch_kv_cache)
            for output_logits, kv_cache, task_req in zip(batch_output_logits, batch_kv_cache, _batch_prefill_request):
                print("拆batch start: ", output_logits, kv_cache, task_req)
                
                _input_ids = output_logits[task_req.batch_input_ids != self._padding_id]
                _attention_mask = task_req.batch_attention_mask[:output_logits.shape[0]]
                _kv_cache = kv_cache[:output_logits.shape[0]]

                print("拆batch: end", _input_ids, _attention_mask, _kv_cache)

                decode_req = DecodeRequest(
                    id=task_req.id,
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    kv_cache=_kv_cache,
                    generated_len=1,
                )
                await self._next_queue.put(decode_req)
    
if __name__ == "__main__":
    
    async def test():
        _PrefillWorker()
        _input_queue = GlobalState.get_prefill_queue()
        request = PrefillRequest(
            id="123",
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

        await _input_queue.put(request)
        _decode_queue = GlobalState.get_decode_queue()
        decode_requrest = await _decode_queue.get()
        print("测试完成，decode request: ", decode_requrest)
    
    asyncio.run(test(), debug=True)
