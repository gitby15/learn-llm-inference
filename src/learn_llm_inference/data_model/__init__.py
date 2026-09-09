from typing import Any, TypedDict
from dataclasses import dataclass
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

@dataclass(slots=True)
class TokenizeRequest:
    meta_info: MetaInfo
    messages: list[ChatMessage]
    


@dataclass(slots=True)
class PrefillRequest:
    meta_info: MetaInfo
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
    
@dataclass(slots=True)
class DecodeRequest:
    meta_info: MetaInfo
    token_id: torch.Tensor # 应该是一个零维标量
    attention_mask: torch.Tensor
    kv_cache: Any
    generated_len: int = 1
    
    

@dataclass(slots=True)
class ResponseRequest:
    meta_info: MetaInfo
    token_id: int
    generated_len: int
    finish_reason: str | None = None


@dataclass(slots=True)
class ResponseChunk:
    id: str
    text: str
    generated_len: int
    finish_reason: str | None = None
