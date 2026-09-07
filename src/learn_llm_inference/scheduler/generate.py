from typing import Any, ClassVar, Self

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from learn_llm_inference.utils import MODEL_ID


class Generator:
    _instance: ClassVar[Self | None] = None
    _model: ClassVar[PreTrainedModel | None] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
            print(f"模型加载完成，Device：{cls._model.device}")
        return cls._instance

    def _get_model(self) -> PreTrainedModel:
        model = self.__class__._model
        if model is None:
            raise RuntimeError("模型未加载")
        return model

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        model = self._get_model()
        return model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=True,
        )

    @torch.inference_mode()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Any,
    ) -> CausalLMOutputWithPast:
        model = self._get_model()
        return model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            past_key_values=kv_cache,
            use_cache=True,
        )
