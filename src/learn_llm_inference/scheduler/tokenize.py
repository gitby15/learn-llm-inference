from typing import TypedDict, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from learn_llm_inference.utils import MODEL_ID
from learn_llm_inference.data_model import ChatMessage


class PrefillInput(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


# Todo: 这里的单例写得有点丑，晚一点优化一下
# Todo: 注意线程安全问题
class Tokenizer:
    _instance = None
    _tokenizer:PreTrainedTokenizerBase
    _eos_id: int

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                local_files_only=True,
            )

            print(f"分词器加载完成: {MODEL_ID}")
            _tokenizer.padding_side = 'left'

            cls._tokenizer = _tokenizer
            cls._eos_id = _tokenizer.eos_token_id

        return cls._instance


    # 私有变量
    def _get_tokenizer(self):
        return self.__class__._tokenizer

    def get_eos_id(self):
        return self.__class__._eos_id
    def get_padding_id(self):
        return self.__class__._tokenizer.pad_token_id
               

    def encode(
        self,
        messages: list[ChatMessage],
    ) -> PrefillInput:
        tokenizer = self._get_tokenizer()
        conversation: list[dict[str, str]] = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]
        result = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        return cast(PrefillInput, result)

    def detokenizer(self, logits: torch.Tensor):
        tokenizer = self._get_tokenizer()
        result = tokenizer.batch_decode(logits, skip_special_tokens=True)
        return result

    def detokenizer_output_only(self, total_logits: torch.Tensor, input_ids: torch.Tensor):
        result = self.detokenizer(total_logits[:, input_ids.shape[-1]:])
        return result
       
