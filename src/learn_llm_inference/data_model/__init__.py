import asyncio
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypedDict

import torch


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(slots=True)
class MetaInfo:
    id: str
    max_tokens: int = 512
    # Todo: 后面再支持这些
    # temperature: float = 0.7
    # top_k: int = 50


@dataclass(frozen=True, slots=True)
class TokenEvent:
    token_id: int
    generated_len: int
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    message: str
    type: str = "server_error"
    code: str = "internal_server_error"
    param: str | None = None

    @classmethod
    def internal(cls) -> "ErrorEvent":
        return cls(message="Internal server error")


ResponseEvent: TypeAlias = TokenEvent | ErrorEvent


@dataclass(slots=True)
class RequestContext:
    meta_info: MetaInfo
    response_queue: asyncio.Queue[ResponseEvent] = field(
        default_factory=asyncio.Queue
    )
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def emit(self, event: ResponseEvent) -> bool:
        if self.cancelled.is_set():
            return False
        await self.response_queue.put(event)
        return True

    async def fail(self) -> bool:
        return await self.emit(ErrorEvent.internal())

    def cancel(self) -> None:
        self.cancelled.set()


@dataclass(slots=True)
class TokenizeRequest:
    context: RequestContext
    messages: list[ChatMessage]


@dataclass(slots=True)
class PrefillRequest:
    context: RequestContext
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(slots=True)
class DecodeRequest:
    context: RequestContext
    token_id: torch.Tensor  # 应该是一个零维标量
    attention_mask: torch.Tensor
    kv_cache: Any
    generated_len: int = 1


@dataclass(slots=True)
class ResponseChunk:
    id: str
    text: str
    generated_len: int
    finish_reason: str | None = None
