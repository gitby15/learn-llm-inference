import asyncio
import logging

from learn_llm_inference.llm_engine.tokenize import Tokenizer
from learn_llm_inference.data_model import TokenizeRequest, PrefillRequest
from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.bridge._utils import BaseWorker

logger = logging.getLogger(__name__)


class _TokenizeWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self._tokenize_queue: asyncio.Queue[TokenizeRequest] = (
            GlobalState.get_tokenize_queue()
        )
        self._prefill_queue: asyncio.Queue[PrefillRequest] = (
            GlobalState.get_prefill_queue()
        )
        self._tokenizer = Tokenizer()

    async def run_forever(self) -> None:
        # tokenize
        while True:
            request = await self._tokenize_queue.get()
            if request.context.cancelled.is_set():
                continue
            try:
                tokenize_result = self._tokenizer.encode(request.messages)
                prefill_req = PrefillRequest(
                    context=request.context,
                    input_ids=tokenize_result["input_ids"],
                    attention_mask=tokenize_result["attention_mask"],
                )
                await self._prefill_queue.put(prefill_req)
            except Exception:
                logger.exception(
                    "Tokenize request %s failed",
                    request.context.meta_info.id,
                )
                await request.context.fail()
