from contextlib import asynccontextmanager

import json
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from learn_llm_inference.data_model import (
    ChatMessage,
    ErrorEvent,
    MetaInfo,
    RequestContext,
    TokenizeRequest,
)
from learn_llm_inference.bridge.bridge_engine import bridge_instance


class OpenAIRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    ignore_eos: bool = False
    # 这俩有一个就行，max_tokens是旧版本，max_completion_tokens是新版本
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)


class ChatChoice_(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage_(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice_]
    usage: ChatUsage_


@asynccontextmanager
async def lifespan(app_: FastAPI):
    app_.state.test = "test"
    await bridge_instance.start()
    try:
        yield
    finally:
        await bridge_instance.stop()
        del app_.state.test


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthy() -> dict[str, str]:
    return {"status": "healthy"}
    

# 为了方便调试，暂时打开GET
@app.post("/v1/chat/completions")
async def openai_api(
    request: OpenAIRequest,
) -> StreamingResponse:
    request_id = f"chatcmpl-{uuid4().hex}"
    context = RequestContext(
        meta_info=MetaInfo(
            id=request_id,
            max_tokens=(
                request.max_completion_tokens or request.max_tokens or 512
            ),
        )
    )
    tokenize_req = TokenizeRequest(
        context=context,
        messages=request.messages,
    )
    response_worker = await bridge_instance.commit_request(tokenize_req)

    async def event_generator():
        try:
            first_chunk = True
            async for event in response_worker.get_stream_response():
                # Todo: 直接把异常raise到外面，是不是简单一些
                if isinstance(event, ErrorEvent):
                    error_payload = {
                        "error": {
                            "message": event.message,
                            "type": event.type,
                            "param": event.param,
                            "code": event.code,
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    break

                delta = {"content": event.text}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                payload = {
                    "id": event.id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": event.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            fallback_error = ErrorEvent.internal()
            error_payload = {
                "error": {
                    "message": fallback_error.message,
                    "type": fallback_error.type,
                    "param": fallback_error.param,
                    "code": fallback_error.code,
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            response_worker.close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
