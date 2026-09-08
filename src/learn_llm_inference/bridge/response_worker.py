import asyncio
from collections.abc import AsyncIterator

from learn_llm_inference.bridge.global_state import GlobalState
from learn_llm_inference.data_model import ResponseChunk
from learn_llm_inference.llm_engine.tokenize import Tokenizer


_STREAM_END = object()


class _ResponseWorker:
    def __init__(self, id: str):
        self._id = id
        self._input_queue = GlobalState.get_response_queue(id)
        self._output_queue: asyncio.Queue[ResponseChunk | Exception | object] = (
            asyncio.Queue()
        )
        self._tokenizer = Tokenizer()
        self._token_cache: list[int] = []
        self._next_generated_len = 1
        self._task = asyncio.create_task(
            self._process_task(),
            name=f"response-worker-{id}",
        )

    async def _process_task(self) -> None:
        while True:
            request = await self._input_queue.get()
            if request.id != self._id:
                raise ValueError(
                    f"expected response for {self._id}, got {request.id}"
                )
            if request.generated_len != self._next_generated_len:
                raise ValueError(
                    "out-of-order response for "
                    f"{self._id}: expected token {self._next_generated_len}, "
                    f"got {request.generated_len}"
                )

            self._token_cache.append(request.token_id)
            text = self._tokenizer.decode(self._token_cache)
            chunk_text = self._take_text(text, finished=request.finished)

            if chunk_text or request.finished:
                await self._output_queue.put(
                    ResponseChunk(
                        id=request.id,
                        text=chunk_text,
                        generated_len=request.generated_len,
                        finished=request.finished,
                        finish_reason=request.finish_reason,
                    )
                )

            if request.finished:
                return
            self._next_generated_len += 1

    def _take_text(self, text: str, *, finished: bool) -> str:
        # 达到输出条件，就输出当前缓存中的文本
        # 输出条件：decode结束了、完成一行、一个单词、一个中文字符
        can_output = (
            finished
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
            if item is _STREAM_END:
                return
            if isinstance(item, Exception):
                raise item
            if not isinstance(item, ResponseChunk):
                raise TypeError(f"unexpected response item: {type(item).__name__}")
            yield item

    @staticmethod
    def _is_cjk_character(codepoint: int) -> bool:
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F
        )
