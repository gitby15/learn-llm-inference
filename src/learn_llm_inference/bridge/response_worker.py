import logging
from collections.abc import AsyncIterator

from learn_llm_inference.data_model import (
    ErrorEvent,
    RequestContext,
    ResponseChunk,
)
from learn_llm_inference.llm_engine.tokenize import Tokenizer


logger = logging.getLogger(__name__)


class _ResponseWorker:
    def __init__(self, context: RequestContext):
        self._context = context
        self._tokenizer = Tokenizer()
        self._token_cache: list[int] = []

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

    async def get_stream_response(
        self,
    ) -> AsyncIterator[ResponseChunk | ErrorEvent]:
        while True:
            event = await self._context.response_queue.get()
            if isinstance(event, ErrorEvent):
                yield event
                return

            try:
                self._token_cache.append(event.token_id)
                text = self._tokenizer.decode(self._token_cache)
                chunk_text = self._take_text(
                    text,
                    finish_reason=event.finish_reason,
                )
            except Exception:
                logger.exception(
                    "Response decoding failed for request %s",
                    self._context.meta_info.id,
                )
                yield ErrorEvent.internal()
                return

            if chunk_text or event.finish_reason is not None:
                yield ResponseChunk(
                    id=self._context.meta_info.id,
                    text=chunk_text,
                    generated_len=event.generated_len,
                    finish_reason=event.finish_reason,
                )

            if event.finish_reason is not None:
                return

    def close(self) -> None:
        self._context.cancel()

    @staticmethod
    def _is_cjk_character(codepoint: int) -> bool:
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F
        )
