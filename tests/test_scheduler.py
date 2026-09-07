import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from learn_llm_inference.scheduler.base_scheduler import (
    TaskIterator,
    TaskRequest,
    WaitingList,
    _finish_reason,
)


class FakeTokenizerBackend:
    eos_token_id = 3
    backend_tokenizer = object()


class FakeDecodeStream:
    def __init__(self, skip_special_tokens=True):
        pass

    def step(self, tokenizer, token_ids):
        pieces = {1: "A", 2: "B", 3: ""}
        return "".join(pieces[token_id] for token_id in token_ids)


class FakeTokenizer:
    backend = FakeTokenizerBackend()

    def tokenize(self, messages):
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def _get_tokenizer(self):
        return self.backend


class FakeGenerator:
    decode_calls = 0

    def prefill(self, input_ids, attention_mask):
        logits = torch.zeros((1, 2, 4))
        logits[0, -1, 1] = 1
        return SimpleNamespace(logits=logits, past_key_values=object())

    def decode(self, input_ids, attention_mask, kv_cache):
        self.__class__.decode_calls += 1
        next_id = 2 if self.decode_calls == 1 else 3
        logits = torch.zeros((1, 1, 4))
        logits[0, -1, next_id] = 1
        return SimpleNamespace(logits=logits, past_key_values=object())


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeGenerator.decode_calls = 0
        self.patches = (
            patch("learn_llm_inference.scheduler.base_scheduler.Tokenizer", FakeTokenizer),
            patch("learn_llm_inference.scheduler.base_scheduler.Generator", FakeGenerator),
            patch(
                "learn_llm_inference.scheduler.base_scheduler.DecodeStream",
                FakeDecodeStream,
            ),
        )
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()

    async def test_request_streams_tokens_until_eos(self):
        waiting_list = WaitingList()
        await waiting_list.start()
        try:
            iterator = TaskIterator(TaskRequest(
                id="request-1",
                payload={
                    "model": "test-model",
                    "messages": [],
                    "max_tokens": 8,
                    "ignore_eos": False,
                },
            ))
            await waiting_list.insert_task(iterator)

            async def collect():
                return [chunk async for chunk in iterator.get_stream_response()]

            chunks = await asyncio.wait_for(collect(), timeout=1)
        finally:
            await waiting_list.stop()

        self.assertEqual(
            [chunk["choices"][0]["delta"].get("content") for chunk in chunks],
            ["A", "B", ""],
        )
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    async def test_max_tokens_one_finishes_without_decode(self):
        waiting_list = WaitingList()
        await waiting_list.start()
        try:
            iterator = TaskIterator(TaskRequest(
                id="request-2",
                payload={
                    "model": "test-model",
                    "messages": [],
                    "max_tokens": 1,
                    "ignore_eos": False,
                },
            ))
            await waiting_list.insert_task(iterator)
            chunk = await asyncio.wait_for(
                anext(iterator.get_stream_response()), timeout=1
            )
        finally:
            await waiting_list.stop()

        self.assertEqual(chunk["choices"][0]["finish_reason"], "length")
        self.assertEqual(FakeGenerator.decode_calls, 0)

    def test_ignore_eos(self):
        self.assertIsNone(_finish_reason(3, 1, 2, True, 3))
        self.assertEqual(_finish_reason(3, 2, 2, True, 3), "length")
