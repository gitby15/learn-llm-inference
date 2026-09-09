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
    _pad_id: int

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
            cls._pad_id = _tokenizer.pad_token_id

        return cls._instance


    # 私有变量
    def _get_tokenizer(self):
        return self.__class__._tokenizer

    def get_eos_id(self):
        return self.__class__._eos_id
    def get_padding_id(self) -> int:
        return self.__class__._pad_id
               

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



    def decode(self, token_ids: list[int]) -> str:
        tokenizer = self._get_tokenizer()
        result = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(result, str):
            return result
        else:
            err_msg = f"tokenizer.decode return {result}, expect str"
            print(err_msg)
            raise ValueError(err_msg)


       
