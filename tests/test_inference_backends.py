"""Tests for the generation backends' multi-sample handling."""

from __future__ import annotations

import sys
import types

from robust_kernelbench.agent.inference_common import APIBackend, VLLMBackend


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "prompt" if kwargs.get("tokenize") is False else [1, 2, 3]


class _RequestOutput:
    def __init__(self, texts):
        self.outputs = [types.SimpleNamespace(text=text) for text in texts]


class _Engine:
    def __init__(self, texts):
        self.texts = texts

    def generate(self, prompts, **kwargs):
        return [_RequestOutput(self.texts)]


def test_vllm_backend_returns_every_sample(monkeypatch):
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = SamplingParams
    fake_request = types.ModuleType("vllm.lora.request")
    fake_request.LoRARequest = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_request)

    backend = VLLMBackend(_Engine([" first ", "second"]), _Tokenizer())
    result = backend.generate([{"role": "user", "content": "hi"}], num_samples=2)

    assert result.generations == ["first", "second"]


def test_api_backend_forwards_n_and_returns_all_choices():
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content="a")),
                    types.SimpleNamespace(message=types.SimpleNamespace(content="b")),
                ],
                usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=7),
            )

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    backend = APIBackend(client, "model")

    result = backend.generate([{"role": "user", "content": "hi"}], num_samples=2)

    assert calls["n"] == 2
    assert result.generations == ["a", "b"]
    assert result.in_tokens == 5
    assert result.out_tokens_raw == 7
