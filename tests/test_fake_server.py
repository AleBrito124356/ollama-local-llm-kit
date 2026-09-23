"""The fake server must speak the real wire protocols, or the rest of the suite proves nothing."""

from __future__ import annotations

import base64
import json
import socket
import struct

import openai
import pytest
import requests

from src.fake_ollama import FakeOllamaServer, hashed_embedding, rag_answer


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_version_tags_and_show(fake):
    assert requests.get(fake.url + "/api/version").json()["version"].endswith("-fake")
    tags = requests.get(fake.url + "/api/tags").json()["models"]
    names = {m["name"] for m in tags}
    assert {"llama3.1:8b", "nomic-embed-text:latest", "bge-m3:latest"} <= names
    show = requests.post(fake.url + "/api/show", json={"model": "bge-m3"}).json()
    assert show["capabilities"] == ["embedding"]
    assert requests.post(fake.url + "/api/show", json={"model": "nope:1b"}).status_code == 404


def test_native_chat_stream_reports_ollama_metrics(fake):
    resp = requests.post(
        fake.url + "/api/chat",
        json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 40}},
        stream=True,
    )
    events = [json.loads(line) for line in resp.iter_lines() if line]
    assert all(not e["done"] for e in events[:-1])
    final = events[-1]
    assert final["done"] and final["eval_count"] == 40 and final["done_reason"] == "length"
    # Canned decode speed for llama3.1:8b is 52.5 tok/s.
    assert final["eval_count"] / (final["eval_duration"] / 1e9) == pytest.approx(52.5, rel=1e-6)
    assert final["load_duration"] == int(2.1e9)  # cold load


def test_openai_sdk_streaming_with_usage(fake):
    client = openai.OpenAI(base_url=fake.v1, api_key="x", max_retries=0)
    stream = client.chat.completions.create(
        model="llama3.2:3b", messages=[{"role": "user", "content": "hello"}],
        stream=True, max_tokens=12, stream_options={"include_usage": True},
    )
    text, usage = "", None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text.startswith("Echo from fake llama3.2:3b: hello")
    assert usage.completion_tokens == 12


def test_openai_sdk_embeddings_base64_roundtrip(fake):
    client = openai.OpenAI(base_url=fake.v1, api_key="x", max_retries=0)
    resp = client.embeddings.create(model="nomic-embed-text", input=["gpu layers", "gpu layer offload"])
    assert len(resp.data) == 2 and len(resp.data[0].embedding) == 768
    raw = requests.post(fake.url + "/v1/embeddings",
                        json={"model": "nomic-embed-text", "input": "x y", "encoding_format": "base64"}).json()
    decoded = struct.unpack("<768f", base64.b64decode(raw["data"][0]["embedding"]))
    assert decoded == pytest.approx(hashed_embedding("x y", 768), abs=1e-6)


def test_hashed_embedding_is_lexically_meaningful():
    q = hashed_embedding("How do I offload layers to the GPU?", 768)
    near = hashed_embedding("Offloading layers onto the GPU with num_gpu", 768)
    far = hashed_embedding("Quantization shrinks weights to four bits", 768)
    assert _cos(q, near) > 0.5 > _cos(q, far)
    # Task prefixes are ignored, exactly like a real model would treat them as instructions.
    assert hashed_embedding("search_query: gpu", 768) == hashed_embedding("gpu", 768)


def test_rag_answer_cites_best_passage():
    prompt = (
        "Context:\n[1] (a.md) Bananas are yellow.\n\n[2] (b.md) Set num_gpu to offload layers to the GPU.\n\n"
        "Question: How do I offload layers?\n\nAnswer:"
    )
    assert rag_answer(prompt) == "Set num_gpu to offload layers to the GPU [2]."


def test_tool_calls_and_json_mode(fake):
    client = openai.OpenAI(base_url=fake.v1, api_key="x", max_retries=0)
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {
        "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
    resp = client.chat.completions.create(
        model="llama3.1:8b", messages=[{"role": "user", "content": "What is (3 + 4) * 5?"}], tools=tools)
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "calculate"
    assert json.loads(call.function.arguments) == {"expression": "(3 + 4) * 5"}
    schema = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
    resp = client.chat.completions.create(
        model="llama3.1:8b", response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "schema:\n" + json.dumps(schema)},
                  {"role": "user", "content": "Extract a person from: Ada Lovelace, 36, London."}])
    assert json.loads(resp.choices[0].message.content) == {"name": "Ada Lovelace", "age": 36}


def test_fault_injection(fake):
    resp = requests.post(fake.url + "/api/chat", json={"model": "oom:llama3.1:8b", "messages": []}, stream=True)
    assert resp.status_code == 200
    assert "error" in json.loads(next(line for line in resp.iter_lines() if line))
    resp = requests.post(fake.url + "/api/chat", json={"model": "http500:llama3.1:8b", "messages": []})
    assert resp.status_code == 500 and "memory" in resp.json()["error"]
    resp = requests.post(fake.url + "/api/chat", json={"model": "bge-m3", "messages": []})
    assert resp.status_code == 400 and "does not support chat" in resp.json()["error"]


def test_pull_and_delete_change_state(fake):
    lines = [json.loads(x) for x in requests.post(fake.url + "/api/pull", json={"model": "qwen2.5:7b"}, stream=True).iter_lines() if x]
    assert lines[-1] == {"status": "success"}
    assert any(e.get("completed") == e.get("total") and e.get("total") for e in lines)
    assert requests.post(fake.url + "/api/show", json={"model": "qwen2.5:7b"}).ok
    assert requests.delete(fake.url + "/api/delete", json={"model": "qwen2.5:7b"}).ok
    assert requests.delete(fake.url + "/api/delete", json={"model": "qwen2.5:7b"}).status_code == 404


def test_nim_mode_is_strict(fake_nim):
    assert requests.get(fake_nim.url + "/api/tags").status_code == 404
    bad = openai.OpenAI(base_url=fake_nim.v1, api_key="nvapi-XXXXXXXXXXXXXXXXXXXXXXXX", max_retries=0)
    with pytest.raises(openai.AuthenticationError):
        bad.models.list()
    good = openai.OpenAI(base_url=fake_nim.v1, api_key="nvapi-good-key-123456789", max_retries=0)
    with pytest.raises(openai.BadRequestError, match="input_type"):
        good.embeddings.create(model="nvidia/nv-embedqa-e5-v5", input=["hello"])
    ok = good.embeddings.create(model="nvidia/nv-embedqa-e5-v5", input=["hello"], extra_body={"input_type": "query"})
    assert len(ok.data[0].embedding) == 1024


def test_network_guard_blocks_remote_hosts():
    with pytest.raises(OSError, match="blocked"):
        socket.create_connection(("integrate.api.nvidia.com", 443), timeout=1)
    with pytest.raises(OSError, match="blocked"):
        socket.create_connection(("8.8.8.8", 53), timeout=1)


def test_server_can_run_standalone_on_any_port():
    with FakeOllamaServer(port=0) as server:
        assert requests.get(server.url + "/").text == "Ollama is running"


def test_realtime_mode_makes_wall_clock_match_the_metrics(fake):
    from src import benchmark as bm

    requests.post(fake.url + "/api/chat", json={"model": "llama3.2:3b", "messages": [], "stream": False})  # load
    fake.state.realtime = True
    r = bm.bench_ollama("llama3.2:3b", "hi", 20)
    assert r.decode_tps == pytest.approx(95.0, rel=1e-6)
    assert 50 < r.e2e_tps < 95  # 20 tokens at ~95 tok/s of wall time, plus a short prefill
    assert r.total_s >= 19 / 95
