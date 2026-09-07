from typing import Any, TypedDict
from dataclasses import dataclass
from pydantic import AliasChoices, BaseModel, Field, field_validator
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
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    kv_cache: Any
    generated_len: int = 1

@dataclass(slots=True)
class DetokenizeRequest:
    id: str
    logits: torch.Tensor
