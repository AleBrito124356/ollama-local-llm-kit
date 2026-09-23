"""Chat REPL: model output is printed literally and never crashes the loop."""

from __future__ import annotations

import re

import pytest

from src import chat
from src.chat import ChatSession, handle_command


class ScriptedClient:
    """Stands in for LLMClient: streams fixed tokens, or raises mid-stream."""

    backend = "ollama"

    def __init__(self, tokens=(), error: Exception | None = None):
        self.tokens = list(tokens)
        self.error = error
        self.calls: list[list[dict]] = []

    def model_for(self, role):
        return "scripted:1b"

    def stream(self, messages, model=None, **_kw):
        self.calls.append(messages)
        yield from self.tokens
        if self.error:
            raise self.error


@pytest.fixture
def session(fake):
    return ChatSession("ollama", system="Be brief.", rag=False)


def test_markup_like_tokens_are_printed_literally(session, capsys):
    """Regression: '[/INST]' raised MarkupError and the except-handler re-raised, killing the REPL."""
    session.client = ScriptedClient(["Sure. ", "[/INST]", " more ", "[bold]", "text", " [red]x[/red]", " :smile:"])
    session.send("hello")
    out = capsys.readouterr().out
    assert "Sure. [/INST] more [bold]text [red]x[/red] :smile:" in out
    assert session.history[-1] == {
        "role": "assistant", "content": "Sure. [/INST] more [bold]text [red]x[/red] :smile:"}


def test_error_messages_with_brackets_do_not_crash(session, capsys):
    session.client = ScriptedClient(["partial "], error=RuntimeError("upstream said [/oops] and [bold]"))
    session.send("hello")  # must not raise
    out = capsys.readouterr().out
    assert "Request failed: upstream said [/oops] and [bold]" in out
    assert session.history == []  # a failed turn is not recorded


def test_history_keeps_original_user_text(session):
    session.client = ScriptedClient(["ok"])
    session.send("first")
    session.send("second")
    sent = session.client.calls[-1]
    assert sent[0] == {"role": "system", "content": "Be brief."}
    assert [m["content"] for m in sent[1:]] == ["first", "ok", "second"]


def test_commands(session, capsys, monkeypatch):
    assert handle_command(session, "/model qwen2.5:7b") is True
    assert session.model == "qwen2.5:7b"
    handle_command(session, "/system Talk like a pirate.")
    assert session.system == "Talk like a pirate." and session.history == []
    handle_command(session, "/rag on")
    assert session.rag is True
    handle_command(session, "/rag maybe")
    handle_command(session, "/backend nim")  # no key configured: friendly message, backend unchanged
    assert session.backend == "ollama"
    handle_command(session, "/frobnicate [x]")
    assert handle_command(session, "/exit") is False
    out = capsys.readouterr().out
    assert "Usage: /rag" in out
    assert "build.nvidia.com" in out
    assert "Unknown command '/frobnicate'" in out


def test_backend_switch_to_nim(session, fake_nim):
    handle_command(session, "/model something")
    handle_command(session, "/backend nim")
    assert session.backend == "nim"
    assert session.model_override is None
    assert session.model == "meta/llama-3.3-70b-instruct"


def test_models_command_lists_installed(session, capsys):
    handle_command(session, "/models")
    out = capsys.readouterr().out
    assert "llama3.1:8b" in out and "bge-m3:latest" in out


def test_rag_turn_grounds_and_cites(fake, capsys):
    s = ChatSession("ollama", system="", rag=True)
    s.send("How do I offload layers to the GPU with num_gpu?")
    out = capsys.readouterr().out
    assert "sources: [1] gpu_offloading.md" in out
    assert re.search(r"\[\d\]", s.history[-1]["content"]), "the grounded answer carries [n] citations"
    # History stores the question, not the context-stuffed prompt.
    assert s.history[0]["content"] == "How do I offload layers to the GPU with num_gpu?"


def test_repl_end_to_end_with_piped_stdin(fake, run_cli):
    script = "\n".join(["hello [/INST] world", "/model llama3.2:3b", "second turn", "/exit"]) + "\n"
    out = run_cli("-m", "src.chat", input=script)
    assert out.returncode == 0, out.stderr
    assert "Echo from fake llama3.1:8b: hello [/INST] world" in out.stdout
    assert "Echo from fake llama3.2:3b: second turn" in out.stdout
    assert "Traceback" not in out.stderr


def test_repl_reports_bad_backend(run_cli):
    out = run_cli("-m", "src.chat", "--backend", "bogus", input="")
    assert out.returncode == 1 and "Unknown LLM_BACKEND" in out.stdout


def test_print_plain_is_literal(capsys):
    chat.print_plain("[link=http://x]y[/link] [/]")
    assert capsys.readouterr().out == "[link=http://x]y[/link] [/]\n"
