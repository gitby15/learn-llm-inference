import asyncio
from typing import cast

from learn_llm_inference.llm_engine.tokenize import Tokenizer
from learn_llm_inference.data_model import TokenizeRequest, PrefillRequest
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.bridge._utils import Worker, worker_lifecycle

@worker_lifecycle
class _TokenizeWorker:
    def __init__(self):
        self._tokenize_queue: asyncio.Queue[TokenizeRequest] = GlobalState.get_tokenize_queue()
        self._prefill_queue: asyncio.Queue[PrefillRequest] = GlobalState.get_prefill_queue()
        self._tokenizer = Tokenizer()

    async def _process_task(self) -> None:
        # tokenize
        while True:
            request = await self._tokenize_queue.get()
            tokenize_result = self._tokenizer.encode(request.messages)
            prefill_req = PrefillRequest(
                id=request.id,
                input_ids = tokenize_result['input_ids'],
                attention_mask = tokenize_result['attention_mask'],
            )
            await self._prefill_queue.put(prefill_req)


if __name__ == "__main__":
    from learn_llm_inference.bridge.bridge_engine import bridge_instance

    async def test():
        worker = cast(Worker, _TokenizeWorker())
        worker.start()
        try:
            request = TokenizeRequest(
                id="123",
                messages=[{"role": "user", "content": "Who are you? Please briefly introduce yourself."}],
            )

            await bridge_instance.commit_request(request)
            prefill_item = await GlobalState.get_prefill_queue().get()
        finally:
            await worker.stop()
    
    asyncio.run(test())
