from learn_llm_inference.data_model import (
    TokenizeRequest,
    DecodeRequest,
    PrefillRequest,
    ResponseRequest,
)
import asyncio


class GlobalState:

    _tokenize_queue: asyncio.Queue[TokenizeRequest] = asyncio.Queue()
    _prefill_queue: asyncio.Queue[PrefillRequest] = asyncio.Queue()
    _decode_queue: asyncio.Queue[DecodeRequest] = asyncio.Queue()
    # detokenize是有状态的，需要保存每个任务的detokenize状态
    _response_map: dict[str, asyncio.Queue[ResponseRequest]] = {}

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
    def get_response_queue(id: str) -> asyncio.Queue[ResponseRequest]:
        result = GlobalState._response_map.get(id, None)
        if result is None:
            print(f"ERROR: response queue not found for id {id}")
            raise KeyError(f"response queue not found for id {id}")
        return result

    @staticmethod
    def register_response_queue(
        id: str,
    ) -> asyncio.Queue[ResponseRequest]:
        if id in GlobalState._response_map:
            raise ValueError(f"response queue already exists for id {id}")
        queue: asyncio.Queue[ResponseRequest] = asyncio.Queue()
        GlobalState._response_map[id] = queue
        return queue

    @staticmethod
    def remove_response_queue(id: str) -> None:
        GlobalState._response_map.pop(id, None)

    @staticmethod
    def get_max_batch() -> int:
        return 5
