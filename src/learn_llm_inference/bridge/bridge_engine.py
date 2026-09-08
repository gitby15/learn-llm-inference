# 作为api server和推理Engine的桥梁
# 做文字和token的桥梁
import asyncio
from typing import cast

from learn_llm_inference.bridge._utils import Worker
from learn_llm_inference.bridge.tokenize_worker import _TokenizeWorker
from learn_llm_inference.bridge.prefill_worker import _PrefillWorker
from learn_llm_inference.bridge.decode_worker import _DecodeWorker
# Todo：Response Worker比较特殊，是每一个api请求生成一个，其他的worker都是单例（未来可以优化成线程池）
from learn_llm_inference.bridge.response_worker import _ResponseWorker

from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import TokenizeRequest



class _BridgeEngine:
    def __init__(self):
        self._workers: tuple[Worker, ...] | None = None

    async def start(self) -> None:
        if self._workers is None:
            self._workers = (
                cast(Worker, _TokenizeWorker()),
                cast(Worker, _PrefillWorker()),
                cast(Worker, _DecodeWorker()),
            )
        for worker in self._workers:
            worker.start()

    async def stop(self) -> None:
        if self._workers is None:
            return
        await asyncio.gather(
            *(worker.stop() for worker in reversed(self._workers))
        )


    async def commit_request(
        self,
        task_req: TokenizeRequest,
    ) -> _ResponseWorker:
        GlobalState.register_response_queue(task_req.id)
        response_worker = _ResponseWorker(task_req.id)
        await GlobalState.get_tokenize_queue().put(task_req)
        return response_worker

bridge_instance = _BridgeEngine()


if __name__ == "__main__":
    async def test():
        await bridge_instance.start()
        try:
            task_req = TokenizeRequest(
                id="test-123",
                messages=[{"role": "user", "content": "番茄炒蛋是怎么做的"}],
            )
            response_worker = await bridge_instance.commit_request(task_req)
            async for chunk in response_worker.get_stream_response():
                print(f"chunk: ${chunk}")
        finally:
            await bridge_instance.stop()

    asyncio.run(test(), debug=True)
