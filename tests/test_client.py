"""The unified client: backend/model resolution, key checks and real HTTP round trips."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import openai
import pytest

from src import client as client_mod
from src.client import (
    LLMClient,
    MissingAPIKey,
    describe_active,
    mask_key,
    nim_base_url,
    nim_key_problem,
    ollama_base_url,
    ollama_host,
    require_nim_key,
    resolve_backend,
    resolve_model,
)
from tests.conftest import TEST_NIM_KEY

REPO = Path(__file__).resolve().parent.parent


# -- configuration ------------------------------------------------------------
def test_backend_resolution(monkeypatch):
    assert resolve_backend() == "ollama"
    monkeypatch.setenv("LLM_BACKEND", " NIM ")
    assert resolve_backend() == "nim"
    assert resolve_backend("ollama") == "ollama"
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND 'bogus'"):
        resolve_backend("bogus")


def test_model_env_overrides_apply_at_call_time(monkeypatch):
    assert resolve_model("chat", "ollama") == "llama3.1:8b"
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    assert resolve_model("chat", "ollama") == "qwen2.5:7b"
    with pytest.raises(KeyError, match="known roles"):
        resolve_model("nope", "ollama")


def test_chat_small_is_small_on_nim_too(monkeypatch):
    assert resolve_model("chat_small", "nim") == "meta/llama-3.1-8b-instruct"
    assert resolve_model("chat_small", "nim") != resolve_model("chat", "nim")
    monkeypatch.setenv("NIM_CHAT_SMALL_MODEL", "meta/llama-3.2-3b-instruct")
    assert resolve_model("chat_small", "nim") == "meta/llama-3.2-3b-instruct"


def test_editing_model_map_is_seen_by_every_module(monkeypatch):
    import src.benchmark
    import src.chat
    import src.rag_local  # noqa: F401

    monkeypatch.setitem(client_mod.MODEL_MAP["chat"], "ollama", "edited:1b")
    session = src.chat.ChatSession("ollama", system="", rag=False)
    assert session.model == "edited:1b"


def test_single_module_identity():
    """Regression: modules used to sys.path-hack `from client import ...`, loading the client twice."""
    for name in ("src.chat", "src.rag_local", "src.benchmark", "src.model_manager"):
        importlib.import_module(name)
    assert "client" not in sys.modules
    import src.chat as chat

    assert chat.MissingAPIKey is client_mod.MissingAPIKey
    assert chat.LLMClient is client_mod.LLMClient


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("127.0.0.1:11500", "http://127.0.0.1:11500"),
        ("0.0.0.0:11500", "http://127.0.0.1:11500"),
        ("https://ollama.lan", "https://ollama.lan:11434"),
        ("myhost", "http://myhost:11434"),
    ],
)
def test_ollama_host_normalisation(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", raw)
    assert ollama_host() == expected
    assert ollama_base_url() == expected + "/v1"


# -- NIM keys ---------------------------------------------------------------------
def _env_example_key() -> str:
    for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.lstrip("# ").strip()
        if line.startswith("NVIDIA_API_KEY="):
            return line.split("=", 1)[1]
    raise AssertionError("no NVIDIA_API_KEY line in .env.example")


def test_placeholder_key_from_env_example_is_rejected(monkeypatch):
    """Regression: `cp .env.example .env` used to make the placeholder count as a real key."""
    monkeypatch.setenv("NVIDIA_API_KEY", _env_example_key())
    with pytest.raises(MissingAPIKey, match="placeholder") as info:
        LLMClient.create("nim")
    assert "build.nvidia.com" in str(info.value)


@pytest.mark.parametrize(
    ("key", "fragment"),
    [
        ("", "not set"),
        ("nvapi-XXXXXXXXXXXXXXXXXXXXXXXX", "placeholder"),
        ("nvapi-...", "placeholder"),
        ("<your key>", "placeholder"),
        ("sk-proj-0123456789abcdefghijkl", "does not start with 'nvapi-'"),
        ("nvapi-short", "malformed"),
        ("nvapi-0123456789 abcdefghij", "malformed"),
    ],
)
def test_nim_key_problems(key, fragment):
    assert fragment in nim_key_problem(key)


def test_valid_key_is_accepted_and_masked(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", TEST_NIM_KEY)
    assert nim_key_problem(TEST_NIM_KEY) is None
    assert require_nim_key() == TEST_NIM_KEY
    masked = mask_key(TEST_NIM_KEY)
    assert masked.startswith("nvapi-") and masked.endswith("ghij") and "0123456789" not in masked


def test_missing_key_raises_signup_hint():
    with pytest.raises(MissingAPIKey, match="build.nvidia.com"):
        LLMClient.create("nim")


def test_self_hosted_nim_needs_no_key(monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1/")
    assert nim_base_url() == "http://localhost:8000/v1"
    assert require_nim_key() == "not-needed-for-self-hosted-nim"
    monkeypatch.setenv("LLM_BACKEND", "nim")
    assert "endpoint=http://localhost:8000/v1" in describe_active()
    assert LLMClient.create().base_url == "http://localhost:8000/v1"


# -- HTTP round trips against the fake -----------------------------------------------
def test_chat_stream_and_no_null_max_tokens(fake):
    c = LLMClient.create("ollama")
    assert c.chat([{"role": "user", "content": "ping"}]) == "Echo from fake llama3.1:8b: ping"
    assert "".join(c.stream([{"role": "user", "content": "pong"}], role="chat_small")).endswith("pong")
    bodies = [r["body"] for r in fake.requests("/v1/chat/completions")]
    assert bodies[1]["model"] == "llama3.2:3b"
    assert all("max_tokens" not in b for b in bodies), "unset max_tokens must be omitted, not sent as null"
    c.chat([{"role": "user", "content": "x"}], max_tokens=5)
    assert fake.requests("/v1/chat/completions")[-1]["body"]["max_tokens"] == 5


def test_raw_chat_returns_tool_calls(fake):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    resp = LLMClient.create().raw_chat([{"role": "user", "content": "Weather in Reykjavik?"}], tools=tools)
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls[0].function.arguments == '{"city": "Reykjavik"}'


def test_local_embed_uses_task_prefixes(fake):
    c = LLMClient.create("ollama")
    docs = c.embed(["GPU layers"], input_type="passage")
    query = c.embed(["GPU layers"], input_type="query")
    assert len(docs[0]) == 768
    assert docs == query  # the fake ignores the prefix, like a model treats it as an instruction
    bodies = [r["body"] for r in fake.requests("/v1/embeddings")]
    assert bodies[0]["input"] == ["search_document: GPU layers"]
    assert bodies[1]["input"] == ["search_query: GPU layers"]
    c.embed(["plain"], model="bge-m3")  # no prefix table for this model
    assert fake.requests("/v1/embeddings")[-1]["body"]["input"] == ["plain"]
    assert c.embed([]) == []
    with pytest.raises(ValueError, match="input_type"):
        c.embed(["x"], input_type="document")


def test_nim_embed_sends_input_type(fake_nim):
    """Regression: NIM's asymmetric embed model rejects requests without input_type."""
    c = LLMClient.create("nim")
    assert len(c.embed(["hello"])[0]) == 1024
    assert len(c.embed(["a passage"], input_type="passage")[0]) == 1024
    bodies = [r["body"] for r in fake_nim.requests("/v1/embeddings")]
    assert bodies[0]["input_type"] == "query"
    assert bodies[1]["input_type"] == "passage"
    assert bodies[1]["input"] == ["a passage"]  # no local-style prefix on NIM
    assert fake_nim.requests("/v1/embeddings")[0]["auth"] == f"Bearer {TEST_NIM_KEY}"


def test_nim_chat_uses_nim_model_ids(fake_nim):
    c = LLMClient.create("nim")
    assert c.chat([{"role": "user", "content": "hi"}]).startswith("Echo from fake meta/llama-3.3-70b-instruct")
    assert c.chat([{"role": "user", "content": "hi"}], role="chat_small").startswith(
        "Echo from fake meta/llama-3.1-8b-instruct")


def test_wrong_nim_key_is_a_clean_auth_error(fake_nim, monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", fake_nim.v1)
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-not-an-nvidia-key-at-all")
    c = LLMClient.create("nim", max_retries=0)  # self-hosted URL: key passed through untouched
    with pytest.raises(openai.AuthenticationError):
        c.chat([{"role": "user", "content": "hi"}])


def test_unreachable_ollama_fails_fast():
    c = LLMClient.create("ollama")
    with pytest.raises(openai.APIConnectionError):
        c.chat([{"role": "user", "content": "hi"}])


def test_client_module_cli(run_cli):
    out = run_cli("-m", "src.client", LLM_BACKEND="ollama")
    assert out.returncode == 0 and "backend=ollama" in out.stdout
    bad = run_cli("-m", "src.client", LLM_BACKEND="bogus")
    assert bad.returncode == 1 and "Unknown LLM_BACKEND" in bad.stderr
