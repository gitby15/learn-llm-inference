from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


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