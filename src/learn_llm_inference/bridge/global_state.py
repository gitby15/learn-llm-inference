import asyncio

from learn_llm_inference.data_model import (
    DecodeRequest,
    PrefillRequest,
    TokenizeRequest,
)


class GlobalState:
    _tokenize_queue: asyncio.Queue[TokenizeRequest] = asyncio.Queue()
    _prefill_queue: asyncio.Queue[PrefillRequest] = asyncio.Queue()
    _decode_queue: asyncio.Queue[DecodeRequest] = asyncio.Queue()

    @staticmethod
    def get_tokenize_queue() -> asyncio.Queue[TokenizeRequest]:
        return GlobalState._tokenize_queue

    @staticmethod
    def get_prefill_queue() -> asyncio.Queue[PrefillRequest]:
        return GlobalState._prefill_queue

    @staticmethod
    def get_decode_queue() -> asyncio.Queue[DecodeRequest]:
        return GlobalState._decode_queue

    @staticmethod
    def get_max_batch() -> int:
        return 5
