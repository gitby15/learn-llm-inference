# 作为api server和推理Engine的桥梁
# 做文字和token的桥梁
import asyncio

from learn_llm_inference.bridge._utils import BaseWorker
from learn_llm_inference.bridge.decode_worker import _DecodeWorker
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.bridge.prefill_worker import _PrefillWorker
from learn_llm_inference.bridge.response_worker import _ResponseWorker
from learn_llm_inference.bridge.tokenize_worker import _TokenizeWorker
from learn_llm_inference.data_model import TokenizeRequest


class _BridgeEngine:
    def __init__(self):
        self._workers: tuple[BaseWorker, ...] | None = None

    async def start(self) -> None:
        if self._workers is None:
            self._workers = (
                _TokenizeWorker(),
                _PrefillWorker(),
                _DecodeWorker(),
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
        response_worker = _ResponseWorker(task_req.context)
        await GlobalState.get_tokenize_queue().put(task_req)
        return response_worker


bridge_instance = _BridgeEngine()
