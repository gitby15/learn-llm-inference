from contextlib import asynccontextmanager
import json
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from learn_llm_inference.data_model import ChatMessage, TokenizeRequest
from learn_llm_inference.bridge.bridge_engine import bridge_instance


class OpenAIRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    ignore_eos: bool = False


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
    try:
        yield
    finally:
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
    http_request: Request,
) -> StreamingResponse:
    id = f"chatcmpl-{uuid4().hex}"
    tokenize_req = TokenizeRequest(
        id=id,
        messages=request.messages,
    )
    response_queue = await bridge_instance.commit_request(tokenize_req)

    async def event_generator():
        async for chunk in await response_queue.get():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
