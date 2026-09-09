# 作为api server和推理Engine的桥梁
# 做文字和token的桥梁
import asyncio
from typing import Callable, cast

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
    ) -> tuple[_ResponseWorker, Callable[[str], None]]:
        GlobalState.register_response_queue(task_req.meta_info.id)
        response_worker = _ResponseWorker(task_req.meta_info.id)
        await GlobalState.get_tokenize_queue().put(task_req)
        def _remove_response_worker(id: str):
            GlobalState.remove_response_queue(id)
        return response_worker, _remove_response_worker


bridge_instance = _BridgeEngine()
