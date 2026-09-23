"""model_manager CLI against the fake Ollama server."""

from __future__ import annotations

import pytest

from src import model_manager as mm
from tests.conftest import DEAD_OLLAMA, REFUSED_OLLAMA


def run(capsys, *argv) -> tuple[int, str]:
    code = mm.main(list(argv))
    return code, capsys.readouterr().out


def test_human_size_matches_ollama_list():
    assert mm.human_size(274_302_450) == "274.3 MB"
    assert mm.human_size(4_920_753_328) == "4.9 GB"
    assert mm.human_size(512) == "512 B"
    assert mm.human_size(3.2e15) == "3200.0 TB"


def test_list(fake, capsys):
    code, out = run(capsys, "list")
    assert code == 0
    for name in ("llama3.1:8b", "qwen2.5:14b", "nomic-embed-text:latest"):
        assert name in out
    assert "4.9 GB" in out and "Q4_K_M" in out


def test_du_counts_shared_digests_once(fake, capsys):
    """Regression: llama3.1:latest and llama3.1:8b share blobs but were summed twice."""
    models = mm.list_models()
    rows, total = mm.disk_usage(models)
    assert total == sum(m["size"] for m in models) - 4_920_753_328
    llama = next(r for r in rows if "llama3.1:8b" in r["names"])
    assert llama["names"] == ["llama3.1:8b", "llama3.1:latest"]

    code, out = run(capsys, "du")
    assert code == 0
    assert "llama3.1:latest" in out and "Total on disk" in out
    assert mm.human_size(total) in out
    assert "1 tag(s) share the same manifest digest" in out


def test_du_with_explicit_duplicate_digest(monkeypatch, fake, capsys):
    digest = "sha256:46e0c1" + "0" * 58
    monkeypatch.setattr(mm, "list_models", lambda: [
        {"name": "llama3.1:latest", "size": 4_920_753_328, "digest": digest},
        {"name": "llama3.1:8b", "size": 4_920_753_328, "digest": digest},
    ])
    code, out = run(capsys, "du")
    assert code == 0
    assert out.count("4.9 GB") == 2  # the row and the total, not 9.8 GB
    assert "9.8 GB" not in out


def test_pull_show_remove_cycle(fake, capsys):
    code, out = run(capsys, "pull", "qwen2.5:7b")
    assert code == 0 and "qwen2.5:7b is ready" in out and "writing manifest" in out
    code, out = run(capsys, "show", "qwen2.5:7b")
    assert code == 0 and "completion, tools" in out and "qwen2" in out
    code, out = run(capsys, "remove", "qwen2.5:7b")
    assert code == 0 and "Removed" in out
    code, out = run(capsys, "show", "qwen2.5:7b")
    assert code == 1 and "not installed locally" in out


def test_pull_unknown_model_reports_server_error(fake, capsys):
    code, out = run(capsys, "pull", "definitely-not-a-model:1b")
    assert code == 1
    assert "Pull failed: pull model manifest: file does not exist" in out


def test_remove_unknown(fake, capsys):
    code, out = run(capsys, "remove", "ghost:1b")
    assert code == 1 and "not installed locally" in out


def test_show_prints_parameters_literally(fake, capsys):
    code, out = run(capsys, "show", "llama3.1:8b")
    assert code == 0 and 'stop "<|eot_id|>"' in out


@pytest.mark.parametrize("command", [["list"], ["du"], ["pull", "x"], ["show", "x"], ["remove", "x"]])
def test_unreachable_server(command, capsys):
    code, out = run(capsys, *command)
    assert code == 1
    assert f"Cannot reach Ollama at {DEAD_OLLAMA}" in out
    assert "ollama serve" in out


def test_connection_refused(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", REFUSED_OLLAMA)
    code, out = run(capsys, "list")
    assert code == 1 and "Cannot reach Ollama at http://127.0.0.1:9" in out


def test_recommend_works_offline(capsys):
    code, out = run(capsys, "recommend")
    assert code == 0
    assert "llama3.1:8b" in out and "nomic-embed-text" in out and "llava:7b" in out
