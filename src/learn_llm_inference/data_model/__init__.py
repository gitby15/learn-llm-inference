from typing import Any, TypedDict
from dataclasses import dataclass
import torch

class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(slots=True)
class TokenizeRequest:
    id: str
    messages: list[ChatMessage]
    # Todo: 未来再支持max_tokens、温度、topK等参数


@dataclass(slots=True)
class PrefillRequest:
    id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
@dataclass(slots=True)
class DecodeRequest:
    id: str
    token_id: torch.Tensor # 应该是一个零维标量
    attention_mask: torch.Tensor
    kv_cache: Any
    generated_len: int = 1

@dataclass(slots=True)
class ResponseRequest:
    id: str
    token_id: int
    generated_len: int
    finished: bool = False
    finish_reason: str | None = None


@dataclass(slots=True)
class ResponseChunk:
    id: str
    text: str
    generated_len: int
    finished: bool = False
    finish_reason: str | None = None
