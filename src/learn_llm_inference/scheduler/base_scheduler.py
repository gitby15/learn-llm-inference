from typing import Any

from learn_llm_inference.data_model import ChatRequest
from learn_llm_inference.scheduler.detokenize import DetokenizeWorker
from learn_llm_inference.scheduler.generate import GenerateWorker
from learn_llm_inference.scheduler.tokenize import TokenizeWorker


class TaskRequest:
    payload: Any
    id: int

    def __init__(self, payload: Any, id: int):
        self.payload = payload
        self.id = id


class StreamIterator:
    def __init__(
        self,
        req: TaskRequest,
        tokenizer: TokenizeWorker,
        generator: GenerateWorker,
        detokenizer: DetokenizeWorker,
    ):
        self.req = req
        self.is_done = False
        self.tokenizer = tokenizer
        self.generator = generator
        self.detokenizer = detokenizer


    def build_chunk(self, content: str, is_first_chunk: bool):
        delta = {"content": content}
        if is_first_chunk:
            delta["role"] = "assistant"
            is_first_chunk = False
        return {
            "id": self.req.id,
            "delta": delta,
            "object": "text_completion.chunk",
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        }

    async def get_stream_response(self):
        request = ChatRequest.model_validate(self.req.payload)
        model_input = await self.tokenizer.tokenize(request)
        is_first_chunk = True
        try:
            async for token_ids in self.generator.generate_stream(
                request.model,
                model_input,
            ):
                content = self.detokenizer.push(
                    self.req.id,
                    request.model,
                    token_ids,
                )
                if content:
                    yield self.build_chunk(content, is_first_chunk)
                    is_first_chunk = False

        except BaseException:
            self.detokenizer.discard(self.req.id)
            raise

        remaining_content = self.detokenizer.finish(self.req.id)
        if remaining_content:
            yield self.build_chunk(remaining_content, is_first_chunk)


# 定义基础调度器
class BaseScheduler:
    def __init__(self):
        self.tokenizer = TokenizeWorker()
        self.generator = GenerateWorker()
        self.detokenizer = DetokenizeWorker()

    def commit_request(self, req: TaskRequest):
        return StreamIterator(
            req,
            self.tokenizer,
            self.generator,
            self.detokenizer,
        )


if __name__ == "__main__":

    async def test():
        scheduler = BaseScheduler()
        payload = {
            "model": "openbmb/MiniCPM5-1B",
            "messages": [{"role": "user", "content": "你好"}],
        }
        req = TaskRequest(payload=payload, id=1)
        stream_iter = scheduler.commit_request(req)
        async for chunk in stream_iter.get_stream_response():
            print(chunk)

    import asyncio

    asyncio.run(test())
