import asyncio
from learn_llm_inference.scheduler.tokenize import Tokenizer
from learn_llm_inference.data_model import TokenizeRequest, PrefillRequest
from learn_llm_inference.bridge.global_state import GlobalState

class _TokenizeWorker:
    def __init__(self):
        self._input_queue: asyncio.Queue[TokenizeRequest] = GlobalState.get_tokenize_queue()
        self._next_queue: asyncio.Queue[PrefillRequest] = GlobalState.get_prefill_queue()
        self._tokenizer = Tokenizer()
        self._task = asyncio.create_task(self._process_task(), name="tokenize-worker")
        

    async def _process_task(self) -> None:
        # tokenize
        while True:
            request = await self._input_queue.get()
            print(f"tokenize 开始: {request}")
            tokenize_result = self._tokenizer.encode(request.messages)
            prefill_req = PrefillRequest(
                id=request.id,
                input_ids = tokenize_result['input_ids'],
                attention_mask = tokenize_result['attention_mask'],
            )
            print(f"tokenize 结束: {prefill_req}")
            await self._next_queue.put(prefill_req)


if __name__ == "__main__":
    async def test():
        _TokenizeWorker()
        _input_queue = GlobalState.get_tokenize_queue()
        request = TokenizeRequest(
            id="123",
            messages=[{"role": "user", "content": "Who are you? Please briefly introduce yourself."}],
        )

        await _input_queue.put(request)
        _prefill_queue = GlobalState.get_prefill_queue()
        prefill_item = await _prefill_queue.get()
        print("prefill request: ", prefill_item)
    
    asyncio.run(test())



