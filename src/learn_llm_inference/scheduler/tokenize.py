from transformers import AutoTokenizer

from learn_llm_inference.data_model import ChatRequest


# Todo: 这里的单例写得有点丑，晚一点优化一下
# Todo: 注意线程安全问题
class TokenizerHolder:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        model_id_list = ["openbmb/MiniCPM5-1B"]
        self.tokenizer_map = {}
        for model_id in model_id_list:
            self.get_tokenizer_(model_id)
        self.__class__._initialized = True

    def get_tokenizer_(self, model_id: str):
        if model_id in self.tokenizer_map:
            return self.tokenizer_map[model_id]

        try:
            self.tokenizer_map[model_id] = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
            )
        except Exception as exc:
            raise ValueError(f"请先手动下载模型：{model_id}") from exc
        return self.tokenizer_map[model_id]


class TokenizeWorker:
    def __init__(self):
        pass

    async def tokenize(self, req: ChatRequest):
        tokenizer = TokenizerHolder().get_tokenizer_(req.model)
        result = tokenizer.apply_chat_template(
            [message.model_dump() for message in req.messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        return result
