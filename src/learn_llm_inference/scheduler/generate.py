import asyncio
from queue import Queue
from threading import Thread
from typing import AsyncIterator
import time
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer


MODEL_ID_LIST = ["openbmb/MiniCPM5-1B"]


class TensorIteratorStreamer(BaseStreamer):
    _END = object()

    def __init__(self, skip_prompt: bool = True):
        self.skip_prompt = skip_prompt
        self.next_tokens_are_prompt = True
        self.queue = Queue()

    def put(self, value: torch.Tensor):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if value.ndim > 1:
            if value.shape[0] != 1:
                raise ValueError("流式生成当前仅支持 batch_size=1")
            value = value[0]
        self.queue.put(value.detach().cpu().reshape(-1))

    def end(self):
        self.queue.put(self._END)

    def fail(self, exc: Exception):
        self.queue.put(exc)
        self.end()

    def get(self):
        return self.queue.get()


class GenerateWorker:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return

        self.model_map = {}
        for model_id in MODEL_ID_LIST:
            self.get_model_(model_id)

        self.__class__._initialized = True

    def get_model_(self, model_id: str):
        if model_id in self.model_map:
            return self.model_map[model_id]

        try:
            print(f"加载模型：{model_id}")
            start_time = time.time()
            self.model_map[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
            print(f"模型加载完成，加载时间：{time.time() - start_time}秒，Device：{self.model_map[model_id].device}")
        except Exception as exc:
            raise ValueError(f"请先手动下载模型：{model_id}") from exc

        return self.model_map[model_id]

    @staticmethod
    def _move_inputs_to_model_device(
        inputs: dict[str, torch.Tensor],
        model: AutoModelForCausalLM,
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    async def generate_stream(
        self,
        model_id: str,
        inputs: dict[str, torch.Tensor],
        **generate_kwargs,
    ) -> AsyncIterator[torch.Tensor]:
        model = self.get_model_(model_id)
        inputs = self._move_inputs_to_model_device(inputs, model)
        streamer = TensorIteratorStreamer(skip_prompt=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": generate_kwargs.pop("max_new_tokens", 1024),
            **generate_kwargs,
        }

        def run_generate():
            try:
                model.generate(**generation_kwargs)
            except Exception as exc:
                streamer.fail(exc)

        thread = Thread(target=run_generate, daemon=True)
        thread.start()

        while True:
            item = await asyncio.to_thread(streamer.get)
            if item is streamer._END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        await asyncio.to_thread(thread.join)
