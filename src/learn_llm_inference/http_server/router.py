import json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import uvicorn
from learn_llm_inference.data_model import ChatRequest
from learn_llm_inference.scheduler.base_scheduler import BaseScheduler, TaskRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
scheduler = BaseScheduler()

# # openai 格式的接口
# @app.api_route("/v1/chat/completions", methods=["GET", "POST"])
# async def openai_api():
#     return {
#         "greet": "ni hao ma??"
#     }


# 为了方便调试，暂时打开GET
@app.post("/v1/chat/completions")
async def simple_chat(request: ChatRequest) -> StreamingResponse:
    task_req = TaskRequest(
        payload=request.model_dump(),
        id=int(time.time() * 1000),
    )
    stream_iter = scheduler.commit_request(task_req)

    async def event_generator():
        async for chunk in stream_iter.get_stream_response():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def launch_server():
    host = '127.0.0.1'
    port = 8888
    uvicorn.run(
        "learn_llm_inference.http_server.router:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=["src"],
    )


def main():
    launch_server()
    

if __name__ == "__main__":
    main()
