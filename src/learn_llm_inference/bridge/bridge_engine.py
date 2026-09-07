# 作为api server和推理Engine的桥梁
# 做文字和token的桥梁
from learn_llm_inference.data_model import TokenizeRequest
from learn_llm_inference.bridge.tokenize_worker import tokenize_worker
from learn_llm_inference.bridge.global_state import GlobalState
import asyncio



class _BridgeEngine:
    def __init__(self):
        pass

    async def start(self):
        pass


    async def commit_request(self, task_req:TokenizeRequest):   
        _id = task_req.id
        response_queue = asyncio.Queue()
        _map = GlobalState.get_detokenize_map()
        _map[_id] = response_queue
        await GlobalState.get_tokenize_queue().put(task_req)
        return response_queue

bridge_instance = _BridgeEngine()

