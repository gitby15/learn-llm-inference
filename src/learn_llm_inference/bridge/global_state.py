from learn_llm_inference.data_model import (
    TokenizeRequest,
    DecodeRequest,
    PrefillRequest,
)
import asyncio


class GlobalState:

    _tokenize_queue: asyncio.Queue[TokenizeRequest] = asyncio.Queue()
    _prefill_queue: asyncio.Queue[PrefillRequest] = asyncio.Queue()
    _decode_queue: asyncio.Queue[DecodeRequest] = asyncio.Queue()
    # detokenize是有状态的，需要保存每个任务的detokenize状态
    _detokenize_map = {}

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
    def get_detokenize_map():
        return GlobalState._detokenize_map

    @staticmethod
    def get_max_batch() -> int:
        return 5
