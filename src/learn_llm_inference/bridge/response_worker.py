import asyncio
from collections.abc import AsyncIterator

from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import ResponseChunk
from learn_llm_inference.llm_engine.tokenize import Tokenizer



class _ResponseWorker:
    def __init__(self, id: str):
        self._id = id
        self._input_queue = GlobalState.get_response_queue(id)
        self._output_queue: asyncio.Queue[ResponseChunk | Exception] = (
            asyncio.Queue()
        )
        self._tokenizer = Tokenizer()
        self._token_cache: list[int] = []
        self._task = asyncio.create_task(
            self._process_task(),
            name=f"response-worker-{id}",
        )

    async def _process_task(self) -> None:
        while True:
            request = await self._input_queue.get()
            if request.meta_info.id != self._id:
                raise ValueError(
                    f"expected response for {self._id}, got {request.meta_info.id}"
                )

            self._token_cache.append(request.token_id)
            text = self._tokenizer.decode(self._token_cache)
            chunk_text = self._take_text(text, finish_reason=request.finish_reason)

            if chunk_text or request.finish_reason is not None:
                await self._output_queue.put(
                    ResponseChunk(
                        id=request.meta_info.id,
                        text=chunk_text,
                        generated_len=request.generated_len,
                        finish_reason=request.finish_reason,
                    )
                )

            if request.finish_reason is not None:
                return
            

    def _take_text(self, text: str, *, finish_reason: str | None) -> str:
        # 达到输出条件，就输出当前缓存中的文本
        # 输出条件：decode结束了、完成一行、一个单词、一个中文字符
        is_finished = finish_reason is not None
        can_output = (
            is_finished
            or text.endswith((" ", "\n"))
            or bool(text and self._is_cjk_character(ord(text[-1])))
        )
        if not can_output:
            return ""

        self._token_cache.clear()
        return text

    async def get_stream_response(self) -> AsyncIterator[ResponseChunk]:
        while True:
            item = await self._output_queue.get()
            if isinstance(item, Exception):
                raise item
            yield item
            if item.finish_reason is not None:
                return

    @staticmethod
    def _is_cjk_character(codepoint: int) -> bool:
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F
        )
