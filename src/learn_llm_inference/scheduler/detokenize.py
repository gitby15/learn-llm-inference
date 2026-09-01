from dataclasses import dataclass, field
from threading import Lock

import torch
from tokenizers.decoders import DecodeStream

from learn_llm_inference.scheduler.tokenize import TokenizerHolder


@dataclass
class DetokenizeState:
    model_id: str
    stream: DecodeStream = field(
        default_factory=lambda: DecodeStream(skip_special_tokens=True)
    )
    token_ids: list[int] = field(default_factory=list)
    emitted_length: int = 0


class DetokenizeWorker:
    def __init__(self):
        self.states: dict[int, DetokenizeState] = {}
        self._lock = Lock()

    def push(
        self,
        request_id: int,
        model_id: str,
        token_ids: torch.Tensor,
    ) -> str:
        tokenizer = TokenizerHolder().get_tokenizer_(model_id)
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is None:
            raise TypeError(f"模型 {model_id} 必须使用 fast tokenizer 才能增量解码")

        with self._lock:
            state = self.states.setdefault(
                request_id,
                DetokenizeState(model_id=model_id),
            )
            if state.model_id != model_id:
                raise ValueError(f"请求 {request_id} 的模型不能在生成期间改变")

            new_token_ids = token_ids.detach().cpu().reshape(-1).tolist()
            state.token_ids.extend(new_token_ids)
            content = state.stream.step(backend_tokenizer, new_token_ids) or ""
            state.emitted_length += len(content)
            return content

    def finish(self, request_id: int) -> str:
        with self._lock:
            state = self.states.pop(request_id, None)
            if state is None or not state.token_ids:
                return ""

            tokenizer = TokenizerHolder().get_tokenizer_(state.model_id)
            text = tokenizer.decode(
                state.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text[state.emitted_length:]

    def discard(self, request_id: int):
        with self._lock:
            self.states.pop(request_id, None)
